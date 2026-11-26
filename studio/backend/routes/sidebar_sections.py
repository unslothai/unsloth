# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Custom sidebar layout in the authenticated account's database."""

from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from auth.authentication import get_current_subject
from storage.sidebar_sections import get_sidebar_layout, save_sidebar_layout

router = APIRouter(dependencies = [Depends(get_current_subject)])
Id = Annotated[str, Field(min_length = 1, max_length = 1024)]
Order = Annotated[list[Id], Field(max_length = 50_000)]


class Section(BaseModel):
    name: Annotated[str, Field(min_length = 1, max_length = 60)]
    sort: Literal["manual", "updated"]
    createdAt: Optional[FiniteFloat] = None
    modifiedAt: Optional[FiniteFloat] = None


class Layout(BaseModel):
    model_config = ConfigDict(extra = "forbid")
    sections: dict[Id, Section] = Field(default_factory = dict)
    sectionByChatId: dict[Id, Id] = Field(default_factory = dict)
    sectionByProjectId: dict[Id, Id] = Field(default_factory = dict)
    sectionByPageId: dict[Id, Id] = Field(default_factory = dict)
    rowOrder: dict[Id, Order] = Field(default_factory = dict)
    sectionOrder: Order = Field(default_factory = list)


class OrderMove(BaseModel):
    sectionId: Optional[Id]
    id: Id
    targetId: Id
    edge: Literal["top", "bottom"]


class LayoutEdit(BaseModel):
    operationId: Annotated[str, Field(min_length = 1, max_length = 128)]
    # A browser with no acknowledged layout is importing its legacy data.
    base: Optional[Layout]
    layout: Layout
    orderMoves: list[OrderMove] = Field(default_factory = list)


@router.get("/sidebar-sections", response_model = Layout, response_model_exclude_none = True)
def get_sidebar_sections():
    return get_sidebar_layout()


@router.post("/sidebar-sections", response_model = Layout, response_model_exclude_none = True)
def sync_sidebar_sections(edit: LayoutEdit):
    return save_sidebar_layout(
        edit.operationId,
        (edit.base or Layout()).model_dump(exclude_none = True),
        edit.layout.model_dump(exclude_none = True),
        importing = edit.base is None,
        order_moves = [move.model_dump() for move in edit.orderMoves],
    )
