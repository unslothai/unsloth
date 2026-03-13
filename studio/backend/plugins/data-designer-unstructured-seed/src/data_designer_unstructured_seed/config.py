# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from data_designer.config.seed_source import SeedSource

from .chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, resolve_chunking


class UnstructuredSeedSource(SeedSource):
    seed_type: Literal["unstructured"] = "unstructured"
    path: str = Field(..., min_length = 1)
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP

    @field_validator("path", mode = "after")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"Unstructured seed path is not a file: {path}")
        return value

    @field_validator("chunk_size", mode = "after")
    @classmethod
    def _validate_chunk_size(cls, value: int) -> int:
        size, _ = resolve_chunking(value, 0)
        return size

    @field_validator("chunk_overlap", mode = "after")
    @classmethod
    def _validate_chunk_overlap(cls, value: int, info) -> int:
        size = info.data.get("chunk_size", cls.model_fields["chunk_size"].default)
        _, overlap = resolve_chunking(size, value)
        return overlap
