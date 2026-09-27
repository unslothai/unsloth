# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Optional

from pydantic import BaseModel, Field, StrictStr


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
    builtin_id: Optional[str] = None
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


class BlenderSettings(BaseModel):
    model_config = {"extra": "forbid"}

    port: int = Field(default = 9876, ge = 1, le = 65535, strict = True)
    blender_path: StrictStr = Field(default = "", pattern = r"^[^\x00]*$")


class BlenderTest(BlenderSettings):
    consent: bool = Field(default = False, strict = True)


class BlenderSetup(BlenderSettings):
    is_enabled: bool
    consent: bool = False


class McpBuiltinResponse(BlenderSettings):
    builtin_id: str = "blender"
    display_name: str = "Blender"
    server_id: Optional[str] = None
    is_enabled: bool = False
    available: bool
    unavailable_reason: Optional[str] = None
    min_blender_version: str


class McpStdioDecodeRequest(BaseModel):
    url: StrictStr


class McpStdioCommand(BaseModel):
    command: StrictStr
    arguments: list[StrictStr] = Field(default_factory = list)


class McpStdioEncodeResponse(BaseModel):
    url: str


class McpServerProbeResult(BaseModel):
    ok: bool
    tool_count: int = 0
    error: Optional[str] = None
    blender_ready: Optional[bool] = None
    blender_error: Optional[str] = None


class McpServerImportRequest(BaseModel):
    # A standard mcpServers JSON config (Claude Desktop / Cursor / Cline / VS Code).
    config: dict


class McpServerImportResult(BaseModel):
    created: list[McpServerResponse] = Field(default_factory = list)
    skipped: list[str] = Field(default_factory = list)
    errors: list[str] = Field(default_factory = list)
