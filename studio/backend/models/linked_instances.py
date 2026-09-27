# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pydantic schemas for the linked Unsloth Studio instances API."""

from typing import Optional

from pydantic import BaseModel, Field


class LinkedInstanceCreate(BaseModel):
    name: str = Field(..., max_length = 32, description = "Short name; models appear as @<name>/<model>")
    base_url: str = Field(..., max_length = 2048, description = "The remote's URL, e.g. its trycloudflare.com address")
    api_key: str = Field(..., min_length = 1, max_length = 512, description = "An API key created on the remote")


class LinkedInstanceUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length = 32)
    base_url: Optional[str] = Field(None, max_length = 2048)
    api_key: Optional[str] = Field(None, max_length = 512)
    allow_tools: Optional[bool] = Field(
        None,
        description = "Let this instance's models run tools. They run THERE, under its settings.",
    )


class LinkedInstance(BaseModel):
    id: str
    name: str
    base_url: str
    allow_tools: bool = False
    created_at: str
    updated_at: str


class LinkedInstanceStatus(BaseModel):
    id: str
    online: bool
    error: Optional[str] = None
    models: list[str] = Field(default_factory = list, description = "Model ids as this server exposes them")
    loaded: list[str] = Field(default_factory = list, description = "The subset currently loaded on the remote")
    latency_ms: Optional[int] = None


class LinkedInstanceGpu(BaseModel):
    name: str
    vram_total_gb: Optional[float] = None
    vram_used_gb: Optional[float] = None
    utilization_pct: Optional[float] = None


class LinkedInstanceInfo(BaseModel):
    """What the remote reports about itself; any field can be missing on older releases."""

    id: str
    online: bool
    error: Optional[str] = None
    version: Optional[str] = None
    install_source: Optional[str] = None
    update_available: bool = False
    latest_version: Optional[str] = None
    platform: Optional[str] = None
    python_version: Optional[str] = None
    device_backend: Optional[str] = None
    torch: Optional[str] = None
    transformers: Optional[str] = None
    cuda: Optional[str] = None
    rocm: Optional[str] = None
    llama_cpp: Optional[str] = None
    gpus: list[LinkedInstanceGpu] = Field(default_factory = list)
    cpu_count: Optional[float] = None
    memory_total_gb: Optional[float] = None
    memory_available_gb: Optional[float] = None
    disk_total_gb: Optional[float] = None
    disk_free_gb: Optional[float] = None
    uptime_seconds: Optional[float] = None
    image_model: Optional[str] = Field(None, description = "Image model loaded on the remote, if any")
