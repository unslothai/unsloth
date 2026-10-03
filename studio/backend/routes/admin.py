# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-only admin panel API: installation summary and the model access policy.

Account lifecycle stays in ``routes.accounts``; loading, unloading and deleting models stays in
the inference and models routes. This router adds only what had no home: the blocklist.
"""

from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from auth import model_policy, policy, storage
from auth.authentication import get_current_subject

router = APIRouter(dependencies = [Depends(get_current_subject), Depends(policy.require_owner)])


class ModelPolicyResponse(BaseModel):
    blocked_models: List[str]


class ModelPolicyRequest(BaseModel):
    blocked_models: List[str] = Field(default_factory = list, max_length = 500)


class AdminOverviewResponse(BaseModel):
    total_accounts: int
    active_accounts: int
    blocked_models: List[str]


@router.get("/overview", response_model = AdminOverviewResponse)
def overview():
    accounts = storage.list_accounts()
    return {
        "total_accounts": len(accounts),
        "active_accounts": sum(1 for a in accounts if a.get("is_active")),
        "blocked_models": model_policy.get_blocked_models(),
    }


@router.get("/model-policy", response_model = ModelPolicyResponse)
def get_model_policy():
    return {"blocked_models": model_policy.get_blocked_models()}


@router.put("/model-policy", response_model = ModelPolicyResponse)
def put_model_policy(payload: ModelPolicyRequest):
    return {"blocked_models": model_policy.set_blocked_models(payload.blocked_models)}
