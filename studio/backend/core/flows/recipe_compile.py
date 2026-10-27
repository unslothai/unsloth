# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Recipe group of column nodes -> the recipe JSON `POST /api/data-recipe/jobs` takes.

The inverse of `recipe_import.import_recipe`. Columns come out in the group's node order; the recipe
engine orders them by their references, so they must not form a cycle. Disabled columns stay out.
"""

from __future__ import annotations

import copy
from typing import Any

from core.flows.recipe_import import COLUMN_TYPE, RECIPE_GRAPH_VERSION, column_references


class RecipeCompileError(ValueError):
    pass


def compile_recipe(graph: dict) -> dict:
    """`{recipe, run}` for a recipe graph; raises RecipeCompileError on a broken graph."""
    if not isinstance(graph, dict) or graph.get("kind") != "recipe_graph":
        raise RecipeCompileError("Not a recipe graph")
    if int(graph.get("version") or 0) > RECIPE_GRAPH_VERSION:
        raise RecipeCompileError("This recipe was made by a newer Unsloth Studio")
    group = graph.get("group") or {}
    group_id = group.get("id")
    columns: list[dict] = []
    for node in graph.get("nodes") or []:
        if node.get("type") != COLUMN_TYPE or node.get("group") != group_id or node.get("disabled"):
            continue
        column = (node.get("params") or {}).get("column")
        if not isinstance(column, dict) or not column.get("name"):
            raise RecipeCompileError(f"Column node {node.get('id')} has no column name")
        columns.append(copy.deepcopy(column))
    if not columns:
        raise RecipeCompileError("A Data Recipe needs at least one column")
    names = [c["name"] for c in columns]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise RecipeCompileError(f"Two columns share the name {duplicates[0]!r}")
    every = set(names)
    reads = {c["name"]: column_references(c, every) for c in columns}
    state: dict[str, int] = {}

    def visit(name: str, path: list[str]) -> None:
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            loop = path[path.index(name) :] + [name]
            raise RecipeCompileError("Columns read each other in a loop: " + " -> ".join(loop))
        state[name] = 1
        for ref in reads[name]:
            visit(ref, path + [name])
        state[name] = 2

    for name in names:
        visit(name, [])
    recipe = copy.deepcopy(group.get("config") or {})
    recipe["columns"] = columns
    return {"recipe": recipe, "run": copy.deepcopy(group.get("run") or {})}


def recipe_graph(doc: dict, group_id: str) -> dict:
    """The recipe graph of one Recipe group in a flow document."""
    group = next((g for g in doc.get("groups") or [] if g.get("id") == group_id), None)
    if group is None or group.get("kind") != "recipe":
        raise RecipeCompileError("This step is not inside a Data Recipe group")
    return {
        "kind": "recipe_graph",
        "version": RECIPE_GRAPH_VERSION,
        "group": group,
        "nodes": [n for n in doc.get("nodes") or [] if n.get("group") == group_id],
    }


def normalise(payload: Any) -> dict:
    """The parts of a saved recipe payload a compiled recipe graph must reproduce."""
    recipe = payload["recipe"] if isinstance(payload, dict) and "recipe" in payload else payload
    run = payload.get("run") if isinstance(payload, dict) and "recipe" in payload else None
    return {"recipe": copy.deepcopy(recipe), "run": copy.deepcopy(run or {})}
