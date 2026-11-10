# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data Recipe fixtures shared by the flows recipe tests: the learning recipes, a recipe group flow, fake job routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ._flows_helpers import RouteCtx

BACKEND = Path(__file__).resolve().parents[1]
LEARNING = BACKEND.parent / "frontend" / "src" / "features" / "data-recipes" / "learning-recipes"
SQL = json.loads((LEARNING / "text-to-sql.json").read_text(encoding = "utf-8"))


@pytest.fixture
def fast_recipe_poll(monkeypatch):
    """Recipe jobs are polled with no wait between status reads."""
    from core.flows.nodes import data
    monkeypatch.setattr(data, "POLL_S", 0)


class RecipeCtx(RouteCtx):
    """A recipe node's ctx: saved recipe r1 in full mode, 10 rows, with capability refusals and model loads recorded."""

    seed = 7

    def __init__(
        self,
        routes,
        params = None,
        inputs = None,
    ):
        super().__init__(routes, inputs = inputs or {}, recipe_id = "r1", mode = "full", rows = 10)
        self.params.update(params or {})
        self.caller = None
        self.refuse: dict[str, str] = {}

    def capability_refusal(
        self,
        cap,
        detail = None,
    ):
        return self.refuse.get(cap)

    async def ensure_model(
        self,
        kind,
        choice,
        options = None,
        primary = False,
    ):
        self.loaded = choice
        self.primary = primary


def recipe_flow(payload = SQL) -> dict:
    """The parsed flow a recipe imports as (a Recipe group)."""
    from core.flows.compiler import parse_document
    from core.flows.recipe_import import flow_from_recipe

    doc, _ = flow_from_recipe(payload, name = "Text to SQL")
    parsed, errors = parse_document(doc)
    assert not errors, errors
    return parsed


def recipe_column(doc: dict, name: str) -> dict:
    from core.flows.recipe_import import COLUMN_TYPE
    return next(
        n
        for n in doc["nodes"]
        if n["type"] == COLUMN_TYPE and n["params"]["column"]["name"] == name
    )


def compiled_recipe(doc: dict) -> dict:
    """What compile_flow hands the group's generate step under RECIPE_PARAM."""
    from core.flows.compiler import RECIPE_PARAM, compile_flow
    from core.flows.recipe_import import GENERATE_TYPE

    plan = compile_flow(doc)
    generate = next(n for n in plan.nodes.values() if n["type"] == GENERATE_TYPE)
    return generate["params"][RECIPE_PARAM]


def group_job_routes(job_ids, *, valid = True):
    """Routes for a recipe group job that validates, starts ``job_ids`` in turn and completes; returns (routes, seen paths)."""
    started = iter(job_ids)
    seen = []

    def routes(method, path, params, body):
        seen.append(path)
        if path == "/api/data-recipe/validate":
            problems = [] if valid else [{"message": "Model alias sql-pro: no endpoint"}]
            return {"valid": valid, "errors": problems}
        if path == "/api/data-recipe/jobs":
            return {"job_id": next(started)}
        if path.endswith("/status"):
            return {"status": "completed", "rows": 2, "artifact_path": "/x/recipes/recipe_sql"}
        if path.endswith("/dataset"):
            return {"dataset": [{"sql_prompt": "Find", "sql": "SELECT 1"}], "total": 1}
        raise AssertionError(path)

    return routes, seen


def saved_recipe_routes(job_ids, artifacts):
    """Routes for the saved recipe r1: each job id starts in turn and completes with its artifact path."""
    started = iter(job_ids)
    saved = {"recipe": {"columns": [{"name": "q", "column_type": "expression", "expr": "x"}]}}

    def routes(method, path, params, body):
        if path == "/api/data-recipe/recipes/r1":
            return {"payload": saved}
        if path == "/api/data-recipe/jobs":
            return {"job_id": next(started)}
        if path.endswith("/status"):
            job = path.split("/")[-2]
            return {"status": "completed", "rows": 3, "artifact_path": artifacts.get(job)}
        if path.endswith("/dataset"):
            return {"dataset": [{"q": 1}], "total": 1}
        raise AssertionError(path)

    return routes
