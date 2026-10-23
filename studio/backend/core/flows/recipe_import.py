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
from typing import Any

import jinja2
from jinja2 import meta, nodes

RECIPE_GRAPH_VERSION = 1
COLUMN_TYPE = "recipe.column"
# Fallback for a template Jinja cannot parse: any name inside a {{ ... }} or {% ... %} block can be a column.
_JINJA_BLOCK = re.compile(r"\{\{(.*?)\}\}|\{%(.*?)%\}", re.S)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_GRID = (320, 180)
_ENV = jinja2.Environment()


def _template_names(text: str) -> list[str]:
    """Names a template reads from its context, in first-mention order; attributes, filters and loop variables are not."""
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
    # Hyphenated column names are kept whole too, in case a prompt means the column and not a subtraction.
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
    # A conditional sampler's rules are Jinja expressions over other columns, as Data Designer reads them.
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
            # A subcategory sampler reads its parent category; a timedelta sampler reads its datetime column.
            for key in ("category", "reference_column_name"):
                parent = sub.get(key)
                if isinstance(parent, str) and parent in names and parent not in seen:
                    seen.append(parent)
    return seen


def import_recipe(payload: Any, *, group_id: str = "recipe") -> dict:
    """A saved recipe payload (`{recipe, run, ui}` or a bare recipe) as a Recipe group fragment."""
    recipe, run, ui = _recipe_parts(payload)
    recipe = copy.deepcopy(recipe)
    columns = recipe.pop("columns")
    positions = {
        str(n.get("id")): [float(n.get("x") or 0), float(n.get("y") or 0)]
        for n in ui.get("nodes") or []
        if isinstance(n, dict) and n.get("id") is not None
    }
    names = {
        c.get("name") for c in columns if isinstance(c, dict) and isinstance(c.get("name"), str)
    }
    nodes: list[dict] = []
    edges: list[dict] = []
    for i, column in enumerate(columns):
        if not isinstance(column, dict):
            raise ValueError(f"Recipe column {i + 1} is not an object")
        name = str(column.get("name") or f"column_{i + 1}")
        node_id = f"{group_id}_col_{i + 1}"
        pos = positions.get(name) or [float(i % 4 * _GRID[0]), float(i // 4 * _GRID[1])]
        nodes.append(
            {
                "id": node_id,
                "type": COLUMN_TYPE,
                "pos": pos,
                "group": group_id,
                "title": name,
                "params": {"column": copy.deepcopy(column)},
            }
        )
    by_name = {n["params"]["column"].get("name"): n["id"] for n in nodes}
    for node in nodes:
        for ref in column_references(node["params"]["column"], names):
            edges.append(
                {
                    "id": f"{by_name[ref]}__{node['id']}",
                    "source": by_name[ref],
                    "target": node["id"],
                    "kind": "reference",
                }
            )
    notes = [
        {
            "id": f"{group_id}_{n['id']}",
            "pos": [float(n.get("x") or 0), float(n.get("y") or 0)],
            "text": str(n.get("markdown") or ""),
            "color": n.get("note_color"),
        }
        for n in ui.get("nodes") or []
        if isinstance(n, dict) and n.get("node_type") == "markdown_note"
    ]
    xs = [n["pos"][0] for n in nodes] or [0.0]
    ys = [n["pos"][1] for n in nodes] or [0.0]
    return {
        "kind": "recipe_graph",
        "version": RECIPE_GRAPH_VERSION,
        "group": {
            "id": group_id,
            "kind": "recipe",
            "label": "Data Recipe",
            "rect": [min(xs) - 40, min(ys) - 60, max(xs) - min(xs) + 360, max(ys) - min(ys) + 240],
            # Everything that is not a column: models, providers, tools, processors, seed, unknown keys.
            "config": recipe,
            "run": copy.deepcopy(run),
        },
        "nodes": nodes,
        "edges": edges,
        "notes": notes,
    }
