# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic schemas for the external LLM providers API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

MAX_JSON_SAFE_INTEGER = 9_007_199_254_740_991


# ── Registry (static provider info) ───────────────────────────────
class ProviderRegistryEntry(BaseModel):
    """A supported provider type with its default configuration."""

    provider_type: str = Field(..., description = "Provider identifier (e.g. 'openai', 'mistral')")
    display_name: str = Field(..., description = "Human-readable provider name")
    base_url: str = Field(..., description = "Default API base URL")
    default_models: list[str] = Field(
        default_factory = list, description = "Well-known model IDs for this provider"
    )

    model_capabilities: dict[str, dict[str, bool]] = Field(default_factory = dict)
    supports_streaming: bool = Field(
        True, description = "Whether this provider supports SSE streaming"
    )
    supports_vision: bool = Field(
        False, description = "Whether this provider supports vision/image input"
    )
    supports_tool_calling: bool = Field(
        False, description = "Whether this provider supports tool/function calling"
    )
    supports_studio_tools: bool = Field(
        False,
        description = "Whether Unsloth runs its own tool loop (search/code/MCP/RAG) against this provider",
    )
    hidden: bool = Field(
        False,
        description = "Backend-only entry; the UI surfaces it via a custom preset, not the dropdown",
    )

    auth_kind: Literal["api_key", "chatgpt_oauth"] = "api_key"
    base_url_editable: bool = True
    model_ids_editable: bool = True
    model_list_mode: Literal["remote", "curated"] = Field(
        "remote",
        description = "remote = fetch /models; curated = huge catalogs — UI uses defaults + manual IDs only",
    )


# ── Provider config CRUD ──────────────────────────────────────────
class ProviderCreate(BaseModel):
    """Request to create a saved provider configuration."""

    provider_type: str = Field(..., description = "Provider type from the registry")
    display_name: str = Field(..., description = "User-chosen label (e.g. 'My OpenAI Key')")
    base_url: Optional[str] = Field(
        None,
        description = "Custom base URL (overrides registry default). Omit to use the default.",
    )
    models: list[str] = Field(
        default_factory = list,
        description = "Enabled model IDs for this connection",
    )
    available_models: list[str] = Field(
        default_factory = list,
        description = "Discovered catalog model IDs last fetched for this connection",
    )
    max_output_tokens: Optional[int] = Field(
        None,
        strict = True,
        ge = 64,
        le = MAX_JSON_SAFE_INTEGER,
        description = "Optional maximum Max Tokens cap for this connection",
    )

    encrypted_api_key: Optional[str] = Field(
        None,
        description = "Optional RSA-encrypted API key to persist for this connection",
    )


class ProviderUpdate(BaseModel):
    """Request to update a saved provider configuration."""

    display_name: Optional[str] = Field(None, description = "New display name")
    base_url: Optional[str] = Field(None, description = "New base URL")
    is_enabled: Optional[bool] = Field(None, description = "Enable or disable this provider")
    models: Optional[list[str]] = Field(None, description = "Enabled model IDs for this connection")
    available_models: Optional[list[str]] = Field(
        None,
        description = "Discovered catalog model IDs last fetched for this connection",
    )
    max_output_tokens: Optional[int] = Field(
        None,
        strict = True,
        ge = 64,
        le = MAX_JSON_SAFE_INTEGER,
        description = "Optional maximum Max Tokens cap for this connection",
    )

    encrypted_api_key: Optional[str] = Field(
        None,
        description = "Optional RSA-encrypted replacement API key; omission preserves the saved key",
    )
    clear_api_key: bool = Field(
        False,
        description = "Explicitly remove the saved API key",
    )


class ProviderCredentialMigration(BaseModel):
    """One-time browser credential migration that never replaces a saved key."""

    encrypted_api_key: str = Field(
        ...,
        min_length = 1,
        description = "RSA-encrypted legacy API key to insert only when no key exists",
    )


class ProviderResponse(BaseModel):
    """A saved provider configuration (returned by list/get endpoints)."""

    id: str = Field(..., description = "Unique provider config ID")
    provider_type: str = Field(..., description = "Provider type (e.g. 'openai')")
    display_name: str = Field(..., description = "User-chosen label")
    base_url: str = Field(..., description = "API base URL")
    is_enabled: bool = Field(True, description = "Whether this provider is enabled")

    has_api_key: bool = Field(False, description = "Whether this caller has a saved API key")

    auth_kind: Literal["api_key", "chatgpt_oauth"] = "api_key"
    auth_status: Literal["disconnected", "connected", "reauthorization_required"] = "disconnected"
    models: list[str] = Field(
        default_factory = list,
        description = "Enabled model IDs for this connection",
    )
    available_models: list[str] = Field(
        default_factory = list,
        description = "Discovered catalog model IDs last fetched for this connection",
    )
    max_output_tokens: Optional[int] = Field(
        None,
        description = "Configured maximum Max Tokens cap for this connection",
    )
    created_at: str = Field(..., description = "ISO 8601 creation timestamp")
    updated_at: str = Field(..., description = "ISO 8601 last-update timestamp")


# ── Model listing ─────────────────────────────────────────────────
class ProviderModelInfo(BaseModel):
    """A model available from an external provider."""

    id: str = Field(..., description = "Model ID as expected by the provider API")
    display_name: str = Field("", description = "Human-readable model name")
    context_length: Optional[int] = Field(None, description = "Maximum context length in tokens")
    owned_by: Optional[str] = Field(None, description = "Model owner/organization")


class ProviderModelsRequest(BaseModel):
    """Request to list models from an external provider."""

    provider_id: Optional[str] = Field(
        None, description = "Saved provider config whose stored key may be used"
    )

    provider_type: str = Field(..., description = "Provider type from the registry")
    encrypted_api_key: Optional[str] = Field(
        None,
        description = "RSA-encrypted, base64-encoded API key (optional for local providers)",
    )
    base_url: Optional[str] = Field(
        None, description = "Custom base URL (overrides registry default)"
    )


# ── Connection testing ────────────────────────────────────────────
class ProviderTestRequest(BaseModel):
    """Request to test connectivity to an external provider."""

    provider_id: Optional[str] = Field(
        None, description = "Saved provider config whose stored key may be used"
    )

    provider_type: str = Field(..., description = "Provider type from the registry")
    encrypted_api_key: Optional[str] = Field(
        None,
        description = "RSA-encrypted, base64-encoded API key (optional for local providers)",
    )
    base_url: Optional[str] = Field(
        None, description = "Custom base URL (overrides registry default)"
    )
    model_id: Optional[str] = Field(
        None, description = "Model ID for providers that need a chat probe"
    )


class ProviderTestResult(BaseModel):
    """Result of a provider connectivity test."""

    success: bool = Field(..., description = "Whether the test succeeded")
    message: str = Field(..., description = "Human-readable result message")
    models_count: Optional[int] = Field(
        None, description = "Number of models found (if test succeeded)"
    )
