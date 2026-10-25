# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Player calls (/invoke) on the GPU lane: where the time goes, and how calls share the loaded chat model."""

import asyncio

import pytest

from core.flows.compiler import parse_document
from core.flows.engine import FlowEngine

from ._flows_helpers import doc, node, run_async


def _engine(handlers: dict) -> FlowEngine:
    eng = FlowEngine()
    eng.handlers = handlers
    eng.mint_key = lambda account: ("sk-unsloth-test", None)
    eng.revoke_key = lambda key_id: None
    return eng


def _doc(*types: str):
    parsed, errors = parse_document(doc([node(t.replace(".", "_"), t) for t in types]))
    assert not errors, errors
    return parsed


async def _invoke(
    eng,
    d,
    entry,
    state = None,
    **kw,
):
    return await eng.invoke(
        app = None, flow_id = "f1", doc = d, entry = entry, inputs = {"state": state or {}}, **kw
    )


@run_async
async def test_an_invoke_reports_its_gpu_wait_first_text_and_stop_time():
    gate = asyncio.Event()

    async def scene(ctx):
        await asyncio.sleep(0.05)
        ctx.text_delta("scene", "Once")
        await gate.wait()
        return {"scene": {}}

    eng = _engine({"story.next_scene": scene})
    d = _doc("story.next_scene")
    timing: dict = {}
    task = asyncio.create_task(_invoke(eng, d, "story_next_scene", timing = timing))
    await asyncio.sleep(0.2)
    gate.set()
    await task
    assert timing["gpu_wait_ms"] >= 0
    assert 40 <= timing["first_text_ms"] <= timing["total_ms"]
    assert "stop_ms" not in timing

    gate.clear()
    stopped: dict = {}
    task = asyncio.create_task(_invoke(eng, d, "story_next_scene", timing = stopped))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert 0 <= stopped["stop_ms"] < 1000
