# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer-split serving status, so the UI can show every machine working."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends

from auth import policy
from auth.authentication import get_current_subject
from core.inference import rpc_serving

router = APIRouter()


# Owner only: the payload carries worker command lines, log tails and host paths.
@router.get("/status", dependencies = [Depends(policy.require_owner)])
async def rpc_serving_status(current_subject: str = Depends(get_current_subject)) -> Dict[str, Any]:
    """Topology, the nodes and their devices, the plan and the arguments it launched with."""
    return rpc_serving.status()
