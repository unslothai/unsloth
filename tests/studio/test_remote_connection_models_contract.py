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
CREDENTIAL_BOOTSTRAP = FRONTEND / "features/credentials/bootstrap.ts"
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
    # The backfill tasks are awaited as a batch, and one failing must not sink the rest.
    # That used to be a literal Promise.allSettled here;
    # it is now settleTasksIfCurrent in features/credentials/reconciliation.ts, which allSettles them AND drops the
    # result when the auth session has moved on.
    # That BOTH hops are awaited
    # that half is run for real in test_provider_backfill_awaits_batch.py.
    flat = " ".join(source.split()).replace("( ", "(")
    assert "settleTasksIfCurrent(backfillTasks" in flat
    helper = RECONCILIATION.read_text(encoding = "utf-8")
    assert "export async function settleTasksIfCurrent" in helper
    # Scoped to the helper's own body.
    # The module also allSettles in runCredentialBootstrap, so a module-wide search stays green when
    # settleTasksIfCurrent is regressed to Promise.all
    body = helper.split("export async function settleTasksIfCurrent", 1)[1]
    body = body.split("\nexport ", 1)[0]
    assert (
        "Promise.allSettled(tasks.map(" in body
    ), "the batch must still settle rather than reject on the first failure"


def test_frontend_sync_preserves_local_provider_options():
    source = SYNC_PROVIDERS.read_text(encoding = "utf-8")
    assert "mergeLocalProviderOptions" in source
    assert "promptCacheTtl" in source
    assert "openaiContainerTtlMinutes" in source


def test_connections_are_hydrated_on_startup():
    """Renamed from test_chat_page_hydrates_connections_on_startup, because the
    chat page is no longer where it happens.

    The provider sync moved out of chat-page.tsx into
    features/credentials/bootstrap.ts, which the ROOT route calls. That is a
    wider guarantee, not a narrower one: connections now hydrate on any entry
    into the app rather than only on the chat page. Asserting the old location
    would fail on a change that improved the thing being asserted, so the
    assertion follows the call to where it went and pins both halves -- the
    bootstrap wires the sync, and something actually runs the bootstrap.

    Both halves match CALL sites, not bare names: an import survives deleting
    the call it feeds, so a name-only assertion passes on a startup that
    hydrates nothing."""
    bootstrap = CREDENTIAL_BOOTSTRAP.read_text(encoding = "utf-8")
    assert "syncExternalProvidersFromBackend(providers" in bootstrap
    root = ROOT_ROUTE.read_text(encoding = "utf-8")
    assert (
        "bootstrapPersistedCredentials()" in root
    ), "nothing calls the credential bootstrap, so no page hydrates connections"
    # The call lives inside CredentialBootstrapGate, so the gate has to be rendered too:
    assert "<CredentialBootstrapGate>" in root
    assert "hydratePersistedSettings()" in CHAT_PAGE.read_text(encoding = "utf-8")


def test_providers_api_sends_models_to_backend():
    source = PROVIDERS_API.read_text(encoding = "utf-8")
    assert "available_models: payload.availableModels" in source
    assert "models: payload.models" in source
