# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The models the Desktop apps cards (Claude Desktop, Codex Desktop) offer in their pickers."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class DesktopAppModel(BaseModel):
    id: str
    loaded: bool = False
    quant: Optional[str] = None


async def available_models() -> list[DesktopAppModel]:
    """Studio's chat catalog for the caller: loaded models first, then every downloaded one (GGUF
    with one row per on-disk quant, and other local weights). The same list, and the same ids,
    POST /models/load accepts, so a pick can always be loaded."""
    from routes.llama_compat import _chat_catalog

    rows = await _chat_catalog()
    models = [
        DesktopAppModel(
            id = row["id"],
            loaded = bool(row.get("loaded")),
            quant = row.get("quant") if isinstance(row.get("quant"), str) else None,
        )
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"]
    ]
    models.sort(key = lambda model: (not model.loaded, model.id.lower()))
    return models
