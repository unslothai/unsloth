# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Optional

from pydantic import BaseModel, Field


class McpServerCreate(BaseModel):
    display_name: str
    url: str
    headers: Optional[dict[str, str]] = None
    is_enabled: bool = True
    use_oauth: bool = False


class McpServerUpdate(BaseModel):
    display_name: Optional[str] = None
    url: Optional[str] = None
    # Absent in request body = leave as-is; null = drop all headers; dict = set.
    headers: Optional[dict[str, str]] = None
    is_enabled: Optional[bool] = None
    use_oauth: Optional[bool] = None


class McpServerResponse(BaseModel):
    id: str
    display_name: str
    url: str
    headers: dict[str, str] = Field(default_factory = dict)
    is_enabled: bool = True
    use_oauth: bool = False
    created_at: str
    updated_at: str


class McpServerTestRequest(BaseModel):
    url: str
    headers: Optional[dict[str, str]] = None
    use_oauth: bool = False


class McpServerProbeResult(BaseModel):
    ok: bool
    tool_count: int = 0
    error: Optional[str] = None


class McpServerImportRequest(BaseModel):
    # A standard mcpServers JSON config (Claude Desktop / Cursor / Cline / VS Code).
    config: dict


class McpServerImportResult(BaseModel):
    created: list[McpServerResponse] = Field(default_factory = list)
    skipped: list[str] = Field(default_factory = list)  # display names skipped as duplicates
    errors: list[str] = Field(default_factory = list)


class McpUiResourceResponse(BaseModel):
    """A ui:// template for the sandboxed frame to render."""

    uri: str
    mime_type: str
    text: str
    # _meta.ui: the CSP domains the host builds the sandbox from, plus hints.
    ui: dict = Field(default_factory = dict)


class McpUiToolCallRequest(BaseModel):
    """A tool call a rendered widget asked the host to make."""

    tool_name: str
    arguments: dict = Field(default_factory = dict)
    # Scopes the stdio session to the conversation that produced the widget.
    thread_id: Optional[str] = None
    session_id: Optional[str] = None


class McpUiToolCallResult(BaseModel):
    content: list[dict] = Field(default_factory = list)
    structured_content: Optional[dict] = None
    is_error: bool = False
    meta: Optional[dict] = None
