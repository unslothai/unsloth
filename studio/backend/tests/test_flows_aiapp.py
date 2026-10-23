# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import pytest

from core.flows.compiler import parse_document
from core.flows.nodes import chat
from core.flows.registry import FlowNodeError

from ._flows_helpers import doc, edge, node, run_async
from .test_flows_engine import BASE, _engine


class Ctx:
    def __init__(
        self,
        params = None,
        inputs = None,
        flow_params = None,
        answer = None,
    ):
        self.params = params or {}
        self.inputs = inputs or {}
        self.flow_params = flow_params or {}
        self.seed = 7
        self.calls = []
        self.json_calls = []
        self.answer = answer

    def on_cancel(self, fn):
        pass

    def check_cancelled(self):
        pass

    async def call(
        self,
        method,
        path,
        json = None,
        **kw,
    ):
        self.calls.append((method, path, json))
        return {
            "choices": [{"message": {"content": "<think>hm</think>Paris."}}],
            "usage": {"total_tokens": 9},
        }

    async def llm_json(self, **kwargs):
        self.json_calls.append(kwargs)
        return self.answer


def _agent(monkeypatch, **params):
    async def loaded(
        ctx,
        kind,
        choice,
        options = None,
    ):
        return None

    monkeypatch.setattr(chat, "ensure_model", loaded)
    ctx = Ctx(params = {"prompt": "Capital of France?", **params})
    out = asyncio.run(chat.agent(ctx))
    return ctx, out


def test_agent_asks_only_for_the_search_tools(monkeypatch):
    ctx, out = _agent(
        monkeypatch, web_search = True, knowledge_base = "kb1", mcp = True, max_tool_calls = 3
    )
    body = ctx.calls[-1][2]
    assert body["enable_tools"] is True
    assert body["enabled_tools"] == ["web_search", "search_knowledge_base"]
    assert body["rag_scope"] == {"kb_id": "kb1"}
    # A stale mcp param from an older save is ignored: flows never send MCP tools.
    assert body["mcp_enabled"] is False and body["max_tool_calls_per_message"] == 3
    # Never left unset: an omitted enabled_tools means every built-in, python and terminal included.
    assert "python" not in body["enabled_tools"] and "terminal" not in body["enabled_tools"]
    assert "permission_mode" not in body and "confirm_tool_calls" not in body
    assert out["text"] == "Paris."
    assert out["steps"]["tools"] == ["web_search", "search_knowledge_base"]


def test_agent_without_tools_names_an_empty_tool_list(monkeypatch):
    ctx, _ = _agent(monkeypatch, web_search = False)
    body = ctx.calls[-1][2]
    assert (
        body["enable_tools"] is False
        and body["enabled_tools"] == []
        and body["mcp_enabled"] is False
    )
    assert "rag_scope" not in body and "max_tool_calls_per_message" not in body
    assert "context_overflow" not in body


def test_agent_puts_history_before_the_new_message(monkeypatch):
    async def loaded(
        ctx,
        kind,
        choice,
        options = None,
    ):
        return None

    monkeypatch.setattr(chat, "ensure_model", loaded)
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "ignore me"},
        {"role": "tool", "content": "nope"},
    ]
    ctx = Ctx(params = {"prompt": "and now?", "system": "Be brief."}, inputs = {"history": history})
    asyncio.run(chat.agent(ctx))
    roles = [m["role"] for m in ctx.calls[-1][2]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    # A resumed chat drops its oldest turns when it outgrows the context instead of failing.
    assert ctx.calls[-1][2]["context_overflow"] == "truncate_oldest"
    assert ctx.calls[-1][2]["context_policy"] == "rolling"


def test_classify_returns_one_of_the_categories():
    ctx = Ctx(
        params = {"categories": ["question", "bug report", "other"]},
        inputs = {"text": "The app crashes on start"},
        answer = {"label": "Bug Report", "confidence": 1.7},
    )
    out = asyncio.run(chat.classify(ctx))
    assert out["label"] == "bug report" and out["confidence"] == 1.0
    assert ctx.json_calls[0]["schema"]["properties"]["label"]["enum"] == [
        "question",
        "bug report",
        "other",
    ]
    with pytest.raises(FlowNodeError):
        asyncio.run(chat.classify(Ctx(params = {"categories": ["only"]}, inputs = {"text": "x"})))


def test_classify_prefers_an_exact_label_over_a_substring():
    ctx = Ctx(
        params = {"categories": ["report", "bug report"]},
        inputs = {"text": "It crashes"},
        answer = {"label": "BUG REPORT", "confidence": 0.9},
    )
    assert asyncio.run(chat.classify(ctx))["label"] == "bug report"


def test_extract_passes_the_schema():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    ctx = Ctx(params = {"json_schema": schema}, inputs = {"text": "I am Ada"}, answer = {"name": "Ada"})
    out = asyncio.run(chat.extract(ctx))
    assert out["json"] == {"name": "Ada"} and ctx.json_calls[0]["schema"] == schema


def test_chat_input_memory_and_output():
    played = Ctx(flow_params = {"message": " hi ", "history": [{"role": "user", "content": "a"}, 3]})
    assert asyncio.run(chat.chat_input(played)) == {
        "message": "hi",
        "history": [{"role": "user", "content": "a"}],
    }
    editor = Ctx(params = {"test_message": "Hello!"})
    assert asyncio.run(chat.chat_input(editor))["message"] == "Hello!"
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(10)]
    kept = asyncio.run(chat.chat_memory(Ctx(params = {"turns": 2}, inputs = {"history": turns})))
    assert [t["content"] for t in kept["history"]] == ["6", "7", "8", "9"]
    assert kept["text"].startswith("User: 6\nAssistant: 7")
    with pytest.raises(FlowNodeError):
        asyncio.run(chat.chat_output(Ctx(inputs = {"reply": "  "})))


@run_async
async def test_the_chat_player_invokes_the_output_with_its_message():
    eng = _engine(
        {
            **BASE,
            "chat.input": chat.chat_input,
            "chat.output": chat.chat_output,
        }
    )

    async def echo(ctx):
        return {
            "text": f"you said {ctx.inputs['prompt']} after {len(ctx.inputs.get('history') or [])} turns"
        }

    eng.handlers["llm.agent"] = echo
    d = doc(
        [node("in", "chat.input"), node("a", "llm.agent"), node("out", "chat.output")],
        [
            edge("in", "message", "a", "prompt"),
            edge("in", "history", "a", "history"),
            edge("a", "text", "out", "reply"),
        ],
    )
    parsed, errors = parse_document(d)
    assert not errors, errors
    history = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    first = await eng.invoke(
        app = None,
        flow_id = "f1",
        doc = parsed,
        entry = "out",
        params = {"message": "hi", "history": history},
    )
    assert first["reply"] == "you said hi after 2 turns"
    second = await eng.invoke(
        app = None, flow_id = "f1", doc = parsed, entry = "out", params = {"message": "again"}
    )
    assert second["reply"] == "you said again after 0 turns"


def test_agent_is_local_and_offline_by_default(monkeypatch):
    kinds = []

    async def loaded(
        ctx,
        kind,
        choice,
        options = None,
    ):
        kinds.append(kind)

    monkeypatch.setattr(chat, "ensure_model", loaded)
    ctx = Ctx(params = {"prompt": "hi"})
    asyncio.run(chat.agent(ctx))
    body = ctx.calls[-1][2]
    assert kinds == ["chat"] and body["model"] == "default"
    assert (
        body["enabled_tools"] == [] and "provider_id" not in body and "external_model" not in body
    )
    from core.flows.specs import SPECS_BY_TYPE

    defaults = {p["id"]: p["default"] for p in SPECS_BY_TYPE["llm.agent"]["params"]}
    assert defaults["web_search"] is False


def test_agent_sends_an_image_to_a_local_vision_model(monkeypatch):
    kinds = []

    async def loaded(
        ctx,
        kind,
        choice,
        options = None,
    ):
        kinds.append((kind, choice))

    async def as_url(ctx, media):
        return "data:image/png;base64,AA=="

    monkeypatch.setattr(chat, "ensure_model", loaded)
    monkeypatch.setattr("core.flows.nodes._common.data_url", as_url)
    chat_model, vision_model = {"repo": "chat-only"}, {"repo": "sees-images"}
    ctx = Ctx(
        params = {"prompt": "what is this?", "model": chat_model, "vision_model": vision_model},
        inputs = {"image": {"kind": "image", "ref": "image:1"}},
    )
    asyncio.run(chat.agent(ctx))
    content = ctx.calls[-1][2]["messages"][-1]["content"]
    # The picture goes to the Vision model, never to the chat-only Model.
    assert kinds == [("vision", vision_model)]
    assert content[1]["image_url"]["url"].startswith("data:image/png")
    asyncio.run(
        chat.agent(Ctx(params = {"prompt": "hi", "model": chat_model, "vision_model": vision_model}))
    )
    assert kinds[-1] == ("chat", chat_model)
