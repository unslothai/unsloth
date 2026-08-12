# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contracts for remote connection model persistence (#7281)."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend/src"
PROVIDERS_API = FRONTEND / "features/chat/api/providers-api.ts"
SYNC_PROVIDERS = FRONTEND / "features/chat/sync-external-providers.ts"
CHAT_PAGE = FRONTEND / "features/chat/chat-page.tsx"
# #8299 moved the settle and the startup hydration out of the two files above: the settle into
# the shared credential reconciliation helper, the hydration into an app-root bootstrap that runs
# for every route rather than only when the chat page mounts.
RECONCILIATION = FRONTEND / "features/credentials/reconciliation.ts"
CREDENTIAL_BOOTSTRAP = FRONTEND / "features/credentials/bootstrap.ts"
APP_ROOT = FRONTEND / "app/routes/__root.tsx"
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
    # The settle itself lives in the shared credential helper now, which adds a staleness check
    # in front of it. Both halves are asserted, so neither a backfill that stops being settled
    # nor a helper that stops settling can pass here.
    assert "settleTasksIfCurrent(backfillTasks" in source
    helper = RECONCILIATION.read_text(encoding = "utf-8")
    assert "export async function settleTasksIfCurrent" in helper
    assert "Promise.allSettled" in helper


def test_frontend_sync_preserves_local_provider_options():
    source = SYNC_PROVIDERS.read_text(encoding = "utf-8")
    assert "mergeLocalProviderOptions" in source
    assert "promptCacheTtl" in source
    assert "openaiContainerTtlMinutes" in source


def test_chat_page_hydrates_connections_on_startup():
    """Connections are hydrated from the backend once a session exists.

    The hydration used to run in the chat page's mount effect and now runs in the app-root
    credential bootstrap, which covers every route rather than only the one page, so that is
    where the contract is asserted. The chat page still hydrates its own persisted settings."""
    bootstrap = CREDENTIAL_BOOTSTRAP.read_text(encoding = "utf-8")
    assert "syncExternalProvidersFromBackend" in bootstrap
    assert "runCredentialBootstrap" in bootstrap
    assert "bootstrapPersistedCredentials" in APP_ROOT.read_text(encoding = "utf-8")
    assert "hydratePersistedSettings()" in CHAT_PAGE.read_text(encoding = "utf-8")


def test_providers_api_sends_models_to_backend():
    source = PROVIDERS_API.read_text(encoding = "utf-8")
    assert "available_models: payload.availableModels" in source
    assert "models: payload.models" in source
