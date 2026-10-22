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
        resident["variant"] = kwargs["variant"]

    async def fake_slots(caller):
        return {"llm": resident["llm"]}

    async def fake_resident(caller):
        return {"llm": resident["llm"]}, {"gguf_variant": resident.get("variant")}

    monkeypatch.setattr(loading, "load_llm", fake_load_llm)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(loading, "_resident", fake_resident)
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
        resident["variant"] = kwargs["variant"]

    async def fake_slots(caller):
        return {"llm": resident["llm"]}

    async def fake_resident(caller):
        return {"llm": resident["llm"]}, {"gguf_variant": resident.get("variant")}

    monkeypatch.setattr(loading, "load_llm", fake_load_llm)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(loading, "_resident", fake_resident)
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


def test_a_variant_loaded_outside_flows_is_seen(monkeypatch):
    repo = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    loads = []
    status = {"gguf_variant": "Q8_0"}

    async def fake_load_llm(caller, ctx, **kwargs):
        loads.append(kwargs["variant"])
        status["gguf_variant"] = kwargs["variant"]

    async def fake_slots(caller):
        return {"llm": [repo]}

    async def fake_resident(caller):
        return {"llm": [repo]}, dict(status)

    monkeypatch.setattr(loading, "load_llm", fake_load_llm)
    monkeypatch.setattr(loading, "resident_slots", fake_slots)
    monkeypatch.setattr(loading, "_resident", fake_resident)
    monkeypatch.setattr(_common, "_caller", lambda ctx: ctx.caller)
    monkeypatch.setattr(_common, "_LOADED_VARIANT", {"llm": (repo.lower(), "q4_k_m")})
    ctx = _Ctx()

    async def run():
        # The Chat page loaded Q8 after Flows loaded Q4: a Q4 node must load Q4 again.
        await _common.ensure_model(ctx, "chat", {"repo": repo, "variant": "Q4_K_M"})
        await _common.ensure_model(ctx, "chat", {"repo": repo, "variant": "Q4_K_M"})

    asyncio.run(run())
    assert loads == ["Q4_K_M"]


def test_the_plain_json_retry_can_be_stopped(monkeypatch):
    from core.flows.registry import FlowNodeError

    async def no_load(ctx, kind, model):
        return None

    monkeypatch.setattr(_common, "ensure_model", no_load)

    class Ctx:
        hooks = []
        sent = []
        cancelled = []

        def on_cancel(self, fn):
            self.hooks.append(fn)

        def log(self, *a, **k):
            pass

        async def call(
            self,
            method,
            path,
            json = None,
            **kw,
        ):
            if path == "/api/inference/cancel":
                self.cancelled.append(json["cancel_id"])
                return {}
            self.sent.append(json["cancel_id"])
            if "response_format" in json:
                raise FlowNodeError("response_format is not supported", status = 400)
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    ctx = Ctx()

    async def run():
        out = await _common.llm_json(
            ctx, model = None, system = "", prompt = "p", schema = {"type": "object"}
        )
        for hook in ctx.hooks:
            await hook()
        return out

    assert asyncio.run(run()) == {"ok": True}
    assert len(set(ctx.sent)) == 2 and set(ctx.sent) <= set(ctx.cancelled)
