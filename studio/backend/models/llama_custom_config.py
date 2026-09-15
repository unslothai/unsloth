# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared wire fields for a selected model's llama.cpp configuration."""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class LlamaCppConfigFields(BaseModel):
    llama_cpp_config: Optional[dict[str, Any]] = Field(
        None,
        description = "Versioned per-model INI configuration, or an explicit managed-mode reset.",
    )

    @field_validator("llama_cpp_config", mode = "before")
    @classmethod
    def validate_llama_cpp_config(cls, value):
        if value is None:
            return None
        from core.inference.llama_custom_config import parse_config_source
        return parse_config_source(value).to_wire()
