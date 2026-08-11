# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contracts for remote connection model persistence (#7281)."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend/src"
PROVIDERS_API = FRONTEND / "features/chat/api/providers-api.ts"
SYNC_PROVIDERS = FRONTEND / "features/chat/sync-external-providers.ts"
CHAT_PAGE = FRONTEND / "features/chat/chat-page.tsx"
RECONCILIATION = FRONTEND / "features/credentials/reconciliation.ts"
BOOTSTRAP = FRONTEND / "features/credentials/bootstrap.ts"
ROOT_ROUTE = FRONTEND / "app/routes/__root.tsx"
PROVIDERS_DB = REPO / "studio/backend/storage/providers_db.py"
PROVIDERS_MODELS = REPO / "studio/backend/models/providers.py"


def test_providers_db_stores_model_json_columns():
    source = PROVIDERS_DB.read_text(encoding = "utf-8")
    assert "models_json" in source
    assert "available_models_json" in source
    assert "ALTER TABLE llm_providers ADD COLUMN models_json" in source


def test_provider_api_schemas_expose_models():
    source = PROVIDERS_MODELS.read_text(encoding = "utf-8")
    assert "models: list[str]" in source
    assert "available_models: list[str]" in source


def test_frontend_sync_prefers_server_models_on_remote_clients():
    source = SYNC_PROVIDERS.read_text(encoding = "utf-8")
    assert "config.models" in source
    assert "config.available_models" in source
    assert "serverModels.length > 0" in source


def test_frontend_sync_backfills_local_models_to_backend():
    source = SYNC_PROVIDERS.read_text(encoding = "utf-8")
    assert "updateProviderConfig" in source
    assert "needsModelBackfill" in source
    # The inline `Promise.allSettled(backfillTasks)` moved behind
    # settleTasksIfCurrent, which adds a staleness guard. Assert the call plus
    # what the helper does, so "every backfill task is settled, one slow one
    # cannot drop the rest" is still pinned rather than taken on trust.
    assert "settleTasksIfCurrent(backfillTasks" in source
    helper = RECONCILIATION.read_text(encoding = "utf-8")
    assert "export async function settleTasksIfCurrent" in helper
    assert "await Promise.allSettled(tasks.map((task) => task()))" in helper


def test_frontend_sync_preserves_local_provider_options():
    source = SYNC_PROVIDERS.read_text(encoding = "utf-8")
    assert "mergeLocalProviderOptions" in source
    assert "promptCacheTtl" in source
    assert "openaiContainerTtlMinutes" in source


def test_connections_hydrate_on_startup():
    # Provider hydration moved off the chat page into the credential bootstrap,
    # which the root route runs, so connections now hydrate app-wide instead of
    # only once chat mounts. Follow it there: asserting on the chat page alone
    # would fail on working code, and dropping the assertion would stop pinning
    # that hydration happens at startup at all.
    # Match the call sites, not the bare names: an import survives deleting the
    # call, so a name-only assertion would pass on a startup that hydrates
    # nothing.
    bootstrap = BOOTSTRAP.read_text(encoding = "utf-8")
    assert "syncExternalProvidersFromBackend(providers" in bootstrap
    assert "bootstrapPersistedCredentials()" in ROOT_ROUTE.read_text(encoding = "utf-8")
    assert "hydratePersistedSettings()" in CHAT_PAGE.read_text(encoding = "utf-8")


def test_providers_api_sends_models_to_backend():
    source = PROVIDERS_API.read_text(encoding = "utf-8")
    assert "available_models: payload.availableModels" in source
    assert "models: payload.models" in source
