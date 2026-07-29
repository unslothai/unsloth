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
USER_ASSETS_API = (
    ROOT / "studio" / "frontend" / "src" / "features" / "user-assets" / "api.ts"
).read_text(encoding = "utf-8")
RECIPE_API = (
    ROOT / "studio" / "frontend" / "src" / "features" / "recipe-studio" / "api" / "index.ts"
).read_text(encoding = "utf-8")
EXECUTION_TRACKER = (
    ROOT
    / "studio"
    / "frontend"
    / "src"
    / "features"
    / "recipe-studio"
    / "executions"
    / "tracker.ts"
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


def test_recipe_playwright_clears_the_durable_migration_claim():
    source = PLAYWRIGHT_RECIPE.read_text(encoding = "utf-8")
    assert 'indexedDB.deleteDatabase("unsloth-user-assets-migration-claims")' in source


def test_completed_execution_clears_a_stale_transport_error():
    assert 'incoming.status === "completed"' in EXECUTIONS_DB
    assert "? null\n        : preferString(current.error, incoming.error)" in EXECUTIONS_DB


def test_legacy_import_has_an_indexeddb_claim_when_web_locks_are_unavailable():
    assert 'typeof navigator.locks === "undefined"' in LEGACY_IMPORT
    assert "const claimed = await claimLegacyBrowserDataWithIndexedDb(owner)" in LEGACY_IMPORT
    assert "return claimDurably()" in LEGACY_IMPORT
    assert (
        'DEVICE_IMPORT_LOCK_NAME,\n      { mode: "exclusive" },\n      claimDurably'
        in LEGACY_IMPORT
    )
    assert "database.transaction(" in LEGACY_IMPORT
    assert '"readwrite"' in LEGACY_IMPORT


def test_recipe_editor_waits_for_authoritative_record_before_rendering():
    assert "getCachedRecipe" not in EDIT_RECIPE_PAGE
    assert 'status: "loading"' in EDIT_RECIPE_PAGE


def test_recipe_editor_refetches_after_a_partial_legacy_import():
    assert "let legacyImportError: unknown = null;" in EDIT_RECIPE_PAGE
    assert "legacyImportError = error;" in EDIT_RECIPE_PAGE
    assert "if (!record && legacyImportError)" in EDIT_RECIPE_PAGE
    assert "if (record && legacyImportError)" in EDIT_RECIPE_PAGE
    assert '"Some local recipes could not be imported"' in EDIT_RECIPE_PAGE


def test_equal_event_snapshots_preserve_current_nonterminal_scalars():
    assert (
        "incomingEvent > currentEvent || incomingTerminal || forwardPolledTransition"
        in EXECUTIONS_DB
    )
    assert "mergeTerminalSnapshots(incoming, current, useIncomingState)" in EXECUTIONS_DB


def test_same_event_terminal_snapshots_accept_later_enrichment():
    assert "incomingEvent === currentEvent && incoming.status !== current.status" in EXECUTIONS_DB
    assert "shouldPreferIncomingTerminalScalars(incomingEvent, currentEvent)" in EXECUTIONS_DB


def test_user_asset_reads_capture_the_starting_auth_subject():
    assert "options.expectedSubjectKey ?? getAuthSubjectKey()" in USER_ASSETS_API
    assert "{ expectedSubjectKey }," in USER_ASSETS_API


def test_job_tracking_reads_are_bound_to_the_execution_owner():
    assert "expectedSubjectKey: string;" in EXECUTION_TRACKER
    assert "getRecipeJobStatus(jobId, { expectedSubjectKey })" in EXECUTION_TRACKER
    assert "getRecipeJobAnalysis(jobId, { expectedSubjectKey })" in EXECUTION_TRACKER
    assert "expectedSubjectKey?: string;" in RECIPE_API
