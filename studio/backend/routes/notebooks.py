# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from auth.authentication import get_current_subject
from utils.notebooks import build_notebook_catalog

router = APIRouter()


class NotebookCatalogEntry(BaseModel):
    id: str
    title: str
    notebook_file: str
    category: str
    featured: bool
    studio_model: Optional[str] = None
    colab_url: str
    github_url: str


class NotebookCatalogResponse(BaseModel):
    notebooks: List[NotebookCatalogEntry]
    categories: List[str]


@router.get("", response_model = NotebookCatalogResponse)
async def list_notebooks(
    q: Optional[str] = Query(None), _current_subject: str = Depends(get_current_subject)
) -> NotebookCatalogResponse:
    normalized_query = q.strip() if isinstance(q, str) and q.strip() else None
    raw_entries = build_notebook_catalog(normalized_query)
    notebooks = [NotebookCatalogEntry.model_validate(entry) for entry in raw_entries]
    categories = sorted({entry.category for entry in notebooks})
    return NotebookCatalogResponse(notebooks = notebooks, categories = categories)
