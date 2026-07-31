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
