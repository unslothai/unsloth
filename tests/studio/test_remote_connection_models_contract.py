# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static contracts for remote connection model persistence (#7281)."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "studio/frontend/src"
PROVIDERS_API = FRONTEND / "features/chat/api/providers-api.ts"
CHAT_PROVIDERS = FRONTEND / "features/chat/chat-providers-dialog.tsx"
PROVIDERS_DB = REPO / "studio/backend/storage/providers_db.py"
PROVIDERS_MODELS = REPO / "studio/backend/models/providers.py"


def test_providers_db_stores_model_json_columns():
    source = PROVIDERS_DB.read_text(encoding = "utf-8")
    assert "models_json" in source
    assert "available_models_json" in source
    assert 'ALTER TABLE llm_providers ADD COLUMN models_json' in source


def test_provider_api_schemas_expose_models():
    source = PROVIDERS_MODELS.read_text(encoding = "utf-8")
    assert "models: list[str]" in source
    assert "available_models: list[str]" in source


def test_frontend_sync_prefers_server_models_on_remote_clients():
    source = CHAT_PROVIDERS.read_text(encoding = "utf-8")
    assert "config.models" in source
    assert "config.available_models" in source
    assert "serverModels.length > 0" in source


def test_providers_api_sends_models_to_backend():
    source = PROVIDERS_API.read_text(encoding = "utf-8")
    assert "available_models: payload.availableModels" in source
    assert "models: payload.models" in source
