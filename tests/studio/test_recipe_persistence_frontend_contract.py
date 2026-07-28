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
LEGACY_IMPORT = (
    ROOT / "studio" / "frontend" / "src" / "features" / "user-assets" / "legacy-import.tsx"
).read_text(encoding = "utf-8")
EDIT_RECIPE_PAGE = (
    ROOT
    / "studio"
    / "frontend"
    / "src"
    / "features"
    / "data-recipes"
    / "pages"
    / "edit-recipe-page.tsx"
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


def test_completed_execution_clears_a_stale_transport_error():
    assert 'incoming.status === "completed"' in EXECUTIONS_DB
    assert "? null\n        : preferString(current.error, incoming.error)" in EXECUTIONS_DB


def test_legacy_import_has_an_indexeddb_claim_when_web_locks_are_unavailable():
    assert 'typeof navigator.locks === "undefined"' in LEGACY_IMPORT
    assert "return claimLegacyBrowserDataWithIndexedDb(owner)" in LEGACY_IMPORT
    assert "database.transaction(" in LEGACY_IMPORT
    assert '"readwrite"' in LEGACY_IMPORT


def test_recipe_editor_waits_for_authoritative_record_before_rendering():
    assert "getCachedRecipe" not in EDIT_RECIPE_PAGE
    assert 'status: "loading"' in EDIT_RECIPE_PAGE
