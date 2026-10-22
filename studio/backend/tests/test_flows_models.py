# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

from core.flows.nodes import _common
from studio_mcp import loading


class _Ctx:
    def __init__(self):
        self.resident = {}
        self.statuses = []
        self.caller = object()

    def set_status(
        self,
        status,
        message = None,
    ):
        self.statuses.append(status)


def test_a_loaded_repo_is_reused_even_when_the_node_names_a_variant(monkeypatch):
    loads = []
    resident = {"llm": []}

    async def fake_load_llm(caller, ctx, **kwargs):
        loads.append(kwargs)
        resident["llm"] = [kwargs["model"]]

    async def fake_slots(caller):
        return dict(resident)

    monkeypatch.setattr(loading, "load_llm", fake_load_llm)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(_common, "_caller", lambda ctx: ctx.caller)
    ctx = _Ctx()
    choice = {"repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "variant": "Q4_K_M"}

    async def run():
        await _common.ensure_model(ctx, "chat", choice)
        await _common.ensure_model(ctx, "chat", choice)
        # Another run or page swaps the chat model: the next step must load ours again.
        resident["llm"] = ["someone/else-GGUF"]
        await _common.ensure_model(ctx, "chat", choice)

    asyncio.run(run())
    assert len(loads) == 2
    assert loads[0]["max_seq_length"] == _common.FLOW_CHAT_CONTEXT


def test_a_different_variant_of_the_loaded_repo_is_loaded(monkeypatch):
    loads = []
    resident = {"llm": []}

    async def fake_load_llm(caller, ctx, **kwargs):
        loads.append(kwargs["variant"])
        resident["llm"] = [kwargs["model"]]

    async def fake_slots(caller):
        return dict(resident)

    monkeypatch.setattr(loading, "load_llm", fake_load_llm)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(_common, "_caller", lambda ctx: ctx.caller)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {})
    ctx = _Ctx()
    repo = "unsloth/Qwen3-4B-Instruct-2507-GGUF"

    async def run():
        await _common.ensure_model(ctx, "chat", {"repo": repo, "variant": "Q4_K_M"})
        await _common.ensure_model(ctx, "chat", {"repo": repo, "variant": "Q8_0"})
        await _common.ensure_model(ctx, "chat", {"repo": repo, "variant": "Q8_0"})

    asyncio.run(run())
    assert loads == ["Q4_K_M", "Q8_0"]
