# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training tools: ``start_training``."""

from __future__ import annotations

from typing import Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from studio_mcp.errors import raise_for_status_field
from studio_mcp.outputs import TrainingStarted
from studio_mcp.tools import WRITES, opt_text, route_json


async def start_training(config: dict[str, Any], kind: Literal["llm"] = "llm") -> TrainingStarted:
    """Start a training job from a TrainingStartRequest-shaped ``config``, the same one the Studio UI sends (list_models with ``model`` gives its training defaults; hf_token for a gated model goes in the config). Returns at once; follow it with studio_status. Studio refuses while another training run or an API inference request is active."""
    payload = await route_json("POST", "/api/train/start", json_body = config)
    # The route refuses some starts with a 200 and status "error".
    raise_for_status_field(payload)
    if not isinstance(payload, dict) or not opt_text(payload.get("job_id")):
        raise ToolError("Studio did not start the training job")
    return TrainingStarted(
        job_id = payload["job_id"],
        status = opt_text(payload.get("status")) or "queued",
        message = opt_text(payload.get("message")),
    )


def register_training(mcp: FastMCP) -> None:
    mcp.tool(start_training, annotations = WRITES)
