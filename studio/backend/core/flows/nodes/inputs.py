# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Input nodes: typed values, seeds, uploaded or Library media, and flow-level inputs."""

from __future__ import annotations

import json
import random
from typing import Any

from core.flows.nodes._common import render
from core.flows.registry import FlowNodeError, NodeContext, handler


def _number(value: Any) -> float | int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        text = str(value).strip()
        return int(text) if text.lstrip("-").isdigit() else float(text)
    except (TypeError, ValueError):
        raise FlowNodeError(f"{value!r} is not a number") from None


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _media(value: Any, kind: str) -> dict:
    if isinstance(value, dict) and value.get("kind") == kind and isinstance(value.get("ref"), str):
        return value
    raise FlowNodeError(f"Pick {'an image' if kind == 'image' else 'an audio clip'} for this node")


@handler("input.text")
async def text(ctx: NodeContext) -> dict:
    return {"text": render(str(ctx.params.get("value") or ""), dict(ctx.flow_params or {}))}


@handler("input.number")
async def number(ctx: NodeContext) -> dict:
    return {"value": _number(ctx.params.get("value") or 0)}


@handler("input.toggle")
async def toggle(ctx: NodeContext) -> dict:
    return {"value": _bool(ctx.params.get("value"))}


@handler("input.select")
async def select(ctx: NodeContext) -> dict:
    options = [str(o) for o in ctx.params.get("options") or []]
    value = ctx.params.get("value")
    if options and value not in options:
        value = options[0]
    return {"value": "" if value is None else str(value)}


@handler("input.seed")
async def seed(ctx: NodeContext) -> dict:
    if ctx.params.get("randomize"):
        return {"seed": random.SystemRandom().randrange(2**31)}
    return {"seed": int(_number(ctx.params.get("value") or 0)) % (2**32)}


@handler("input.image")
async def image(ctx: NodeContext) -> dict:
    return {"image": _media(ctx.params.get("asset"), "image")}


@handler("input.audio")
async def audio(ctx: NodeContext) -> dict:
    return {"audio": _media(ctx.params.get("asset"), "audio")}


def coerce_param(kind: str, value: Any) -> Any:
    if value is None:
        return None
    if kind == "number":
        return _number(value)
    if kind == "bool":
        return _bool(value)
    if kind == "json":
        if isinstance(value, str):
            try:
                return json.loads(value)
            except ValueError:
                raise FlowNodeError("The flow input is not valid JSON") from None
        return value
    if kind in ("image", "audio"):
        return _media(value, kind)
    return value if isinstance(value, str) else json.dumps(value)


@handler("input.param")
async def param(ctx: NodeContext) -> dict:
    name = str(ctx.params.get("name") or "input")
    kind = str(ctx.params.get("type") or "text")
    supplied = (ctx.flow_params or {}).get(name)
    value = supplied if supplied is not None else ctx.params.get("default")
    if value in (None, "") and kind in ("image", "audio"):
        raise FlowNodeError(
            f"The flow input {name!r} needs {'an image' if kind == 'image' else 'an audio clip'}"
        )
    return {"value": coerce_param(kind, value if value is not None else "")}


@handler("input.list")
async def items(ctx: NodeContext) -> dict:
    return {"items": [str(i) for i in ctx.params.get("items") or [] if str(i).strip()]}
