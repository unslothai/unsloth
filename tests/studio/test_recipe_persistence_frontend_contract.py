# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXECUTIONS_DB = (
    ROOT
    / "studio"
    / "frontend"
    / "src"
    / "features"
    / "recipe-studio"
    / "data"
    / "executions-db.ts"
).read_text(encoding = "utf-8")
RECIPES_DB = (
    ROOT / "studio" / "frontend" / "src" / "features" / "data-recipes" / "data" / "recipes-db.ts"
).read_text(encoding = "utf-8")
PLAYWRIGHT_RECIPE = ROOT / "tests" / "studio" / "playwright_recipe_persistence.py"


def test_first_job_id_snapshot_bypasses_execution_write_debounce():
    assert "const persistedKey = executionKey(" in EXECUTIONS_DB
    assert "persistedExecutions.get(persistedKey)?.jobId" in EXECUTIONS_DB
    assert "const jobAttached = Boolean(execution.jobId) && !previousJobId;" in EXECUTIONS_DB
    assert "if ((terminal || jobAttached) && state.timer)" in EXECUTIONS_DB
    assert "if (terminal || jobAttached) void drainWrites(key, state);" in EXECUTIONS_DB


def test_recipe_sanitizer_defines_arbitrary_json_keys_as_own_properties():
    assert "Object.defineProperty(output, key" in RECIPES_DB
    assert "enumerable: true" in RECIPES_DB
    assert "output[key] = sanitizeRecipeForPersistence" not in RECIPES_DB


def test_recipe_playwright_does_not_import_jwt_dependent_auth_package():
    tree = ast.parse(PLAYWRIGHT_RECIPE.read_text(encoding = "utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module == "auth" for node in ast.walk(tree)
    )
