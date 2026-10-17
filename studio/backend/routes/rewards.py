# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr

from auth.authentication import get_current_subject
from core.training.rewards import (
    MAX_REWARD_MD_BYTES,
    RewardError,
    RewardExistsError,
    RewardNotFoundError,
    delete_reward,
    export_reward,
    get_reward,
    import_reward,
    list_rewards,
    parse_reward_markdown,
    preview_scores,
)


router = APIRouter()


class RewardRecord(BaseModel):
    name: str
    kind: Literal["rule", "python"] = "rule"
    description: str = ""
    source: Literal["user", "bundled"]
    valid: bool
    shadowed: bool = False
    error: Optional[str] = None
    rule: Optional[dict[str, Any]] = None


class RewardImportRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    markdown: StrictStr = Field(..., max_length = MAX_REWARD_MD_BYTES)
    overwrite: StrictBool = False


class RewardExport(BaseModel):
    name: str
    markdown: str


class RewardPreviewItem(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: Optional[StrictStr] = None
    markdown: Optional[StrictStr] = Field(None, max_length = MAX_REWARD_MD_BYTES)
    weight: float = Field(1.0, ge = -10, le = 10, allow_inf_nan = False)


class RewardPreviewRequest(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    rewards: list[RewardPreviewItem] = Field(..., min_length = 1, max_length = 16)
    completion: StrictStr = Field(..., max_length = 64 * 1024)
    reference: Optional[StrictStr] = Field(None, max_length = 16 * 1024)


class RewardPreviewScore(BaseModel):
    name: str
    score: float
    weighted: float


class RewardPreviewResponse(BaseModel):
    scores: list[RewardPreviewScore]
    total: float


def _http_error(exc: RewardError) -> HTTPException:
    if isinstance(exc, RewardNotFoundError):
        return HTTPException(status_code = 404, detail = str(exc))
    if isinstance(exc, RewardExistsError):
        return HTTPException(status_code = 409, detail = str(exc))
    return HTTPException(status_code = 400, detail = str(exc))


@router.get("", response_model = list[RewardRecord])
def get_rewards(current_subject: str = Depends(get_current_subject)) -> list[dict[str, Any]]:
    try:
        return list_rewards()
    except OSError as exc:
        raise HTTPException(status_code = 500, detail = "Could not read the reward library.") from exc


@router.post("", response_model = RewardRecord, status_code = 201)
def import_reward_route(
    payload: RewardImportRequest, current_subject: str = Depends(get_current_subject)
) -> dict[str, Any]:
    try:
        return import_reward(payload.markdown, overwrite = payload.overwrite)
    except RewardError as exc:
        raise _http_error(exc) from exc


@router.post("/preview", response_model = RewardPreviewResponse)
def preview_rewards(
    payload: RewardPreviewRequest, current_subject: str = Depends(get_current_subject)
) -> dict[str, Any]:
    scores = []
    try:
        for item in payload.rewards:
            if item.markdown is not None:
                spec = parse_reward_markdown(item.markdown)
            elif item.name:
                spec = get_reward(item.name)
            else:
                raise RewardError("Each preview item needs a name or markdown.")
            [result] = preview_scores([spec], payload.completion, payload.reference)
            scores.append({**result, "weighted": result["score"] * item.weight})
    except RewardError as exc:
        raise _http_error(exc) from exc
    return {"scores": scores, "total": sum(s["weighted"] for s in scores)}


@router.get("/{name}/export", response_model = RewardExport)
def export_reward_route(
    name: str, current_subject: str = Depends(get_current_subject)
) -> dict[str, Any]:
    try:
        return {"name": name, "markdown": export_reward(name)}
    except RewardError as exc:
        raise _http_error(exc) from exc


@router.delete("/{name}", status_code = 204, response_class = Response)
def delete_reward_route(name: str, current_subject: str = Depends(get_current_subject)) -> Response:
    try:
        delete_reward(name)
    except RewardError as exc:
        raise _http_error(exc) from exc
    return Response(status_code = 204)
