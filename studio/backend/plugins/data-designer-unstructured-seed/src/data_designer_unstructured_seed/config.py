# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from data_designer.config.seed_source import SeedSource

from .chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, resolve_chunking


class UnstructuredSeedSource(SeedSource):
    seed_type: Literal["unstructured"] = "unstructured"
    paths: list[str] = Field(min_length = 1)

    @model_validator(mode = "before")
    @classmethod
    def _normalize_legacy_path(cls, data):
        if isinstance(data, dict) and "paths" not in data and data.get("path"):
            data = dict(data)
            data["paths"] = [data["path"]]
        return data

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    @field_validator("paths")
    @classmethod
    def _validate_paths(cls, v: list[str]) -> list[str]:
        for p in v:
            expanded = Path(p).expanduser()
            if not expanded.is_file():
                raise ValueError(f"Seed file does not exist: {expanded}")
        return v

    @field_validator("chunk_size")
    @classmethod
    def _resolve_chunk_size(cls, v: int) -> int:
        cs, _ = resolve_chunking(v, 0)
        return cs

    @field_validator("chunk_overlap")
    @classmethod
    def _resolve_chunk_overlap(cls, v: int, info) -> int:
        cs = info.data.get("chunk_size", DEFAULT_CHUNK_SIZE)
        _, co = resolve_chunking(cs, v)
        return co
