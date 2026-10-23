# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Chat and agent nodes: an LLM with Studio's search tools, classify and extract, and the chat player's blocks."""

from __future__ import annotations

import uuid
from typing import Any

from core.flows.nodes._common import (
    _images_content,
    _messages,
    _message_text,
    _register_chat_cancel,
    as_text,
    ensure_model,
    render,
    strip_think,
    template_vars,
)
from core.flows.nodes.text import _prompt, _schema
from core.flows.registry import FlowNodeError, NodeContext, handler

# Only Studio's read-only search tools: a shared flow must never run code or edit files on the importer's machine.
AGENT_TOOLS = ("web_search", "search_knowledge_base")
MAX_HISTORY_TURNS = 50
MAX_TURN_CHARS = 20000


def chat_history(value: Any) -> list[dict]:
    """A conversation as [{role, content}] with only user and assistant turns, newest last and bounded."""
    turns: list[dict] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            continue
        content = as_text(item.get("content"))[:MAX_TURN_CHARS]
        if content:
            turns.append({"role": item["role"], "content": content})
    return turns[-MAX_HISTORY_TURNS:]


def agent_body(ctx: NodeContext, prompt: Any, system: str, history: list[dict]) -> dict:
    """The /v1/chat/completions request to the loaded local model: the chat route applies its own tool rules to the
    run's key, as for any API client; a flow only ever asks for the search tools."""
    messages = _messages(system, prompt)
    messages[-1:-1] = history
    tools = []
    if ctx.params.get("web_search"):
        tools.append("web_search")
    kb = str(ctx.params.get("knowledge_base") or "").strip()
    if kb:
        tools.append("search_knowledge_base")
    body: dict[str, Any] = {
        "model": "default",
        "messages": messages,
        "stream": False,
        "cancel_id": f"flow-{uuid.uuid4().hex[:24]}",
        "temperature": float(
            0.6 if ctx.params.get("temperature") is None else ctx.params["temperature"]
        ),
        "max_tokens": int(ctx.params.get("max_tokens") or 2048),
        # An omitted enabled_tools means every built-in, python and terminal included: always name the list.
        "enable_tools": bool(tools),
        "enabled_tools": [t for t in tools if t in AGENT_TOOLS],
        # MCP tools can write or run commands and this request has no approval prompt, so flows never send them.
        "mcp_enabled": False,
    }
    if history:
        # A resumed chat can outgrow the loaded context: drop its oldest turns instead of failing the reply.
        body["context_overflow"] = "truncate_oldest"
        body["context_policy"] = "rolling"
    if tools:
        body["max_tool_calls_per_message"] = int(ctx.params.get("max_tool_calls") or 8)
    if kb:
        body["rag_scope"] = {"kb_id": kb}
    return body


@handler("llm.agent")
async def agent(ctx: NodeContext) -> dict:
    prompt = _prompt(ctx, "prompt", "prompt")
    if not prompt:
        raise FlowNodeError("The prompt is empty: type one or connect text to Prompt")
    system = render(str(ctx.params.get("system") or ""), template_vars(ctx))
    image = ctx.inputs.get("image")
    # A picture goes to the Vision model (or the vision model already loaded); a chat-only model cannot see it.
    if image:
        await ensure_model(ctx, "vision", ctx.params.get("vision_model"))
    else:
        await ensure_model(ctx, "chat", ctx.params.get("model"))
    content = await _images_content(ctx, prompt, [image] if image else None)
    body = agent_body(ctx, content, system, chat_history(ctx.inputs.get("history")))
    _register_chat_cancel(ctx, body["cancel_id"])
    ctx.check_cancelled()
    payload = await ctx.call("POST", "/v1/chat/completions", json = body)
    text = strip_think(_message_text(payload))
    # The tool loop runs inside the route and the non-streaming answer carries no trace of its calls, so steps says
    # which tools were offered, not which ran.
    steps = {
        "tools": body["enabled_tools"],
        "knowledge_base": (body.get("rag_scope") or {}).get("kb_id"),
        "usage": payload.get("usage") if isinstance(payload, dict) else None,
    }
    return {"text": text, "steps": steps}


@handler("llm.classify")
async def classify(ctx: NodeContext) -> dict:
    text = as_text(ctx.inputs.get("text")).strip()
    if not text:
        raise FlowNodeError("Connect the text to classify")
    labels = [str(c).strip() for c in ctx.params.get("categories") or [] if str(c).strip()]
    if len(labels) < 2:
        raise FlowNodeError("Add at least two categories")
    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": labels},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["label", "confidence"],
    }
    extra = str(ctx.params.get("instructions") or "").strip()
    system = (
        "Classify the text into exactly one of the categories. Give your confidence from 0 to 1."
    )
    if extra:
        system = f"{system}\n{extra}"
    value = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = system,
        prompt = f"Categories: {', '.join(labels)}\n\nText:\n{text}",
        schema = schema,
        temperature = 0.0,
        max_tokens = 256,
        seed = ctx.seed,
    )
    label = value.get("label") if isinstance(value, dict) else None
    if label not in labels:
        # A model without structured output can answer in another case or with extra words: an exact match in any
        # case wins over a category that merely appears inside the answer.
        lowered = str(label or "").strip().lower()
        label = next((c for c in labels if c.lower() == lowered), None) or next(
            (c for c in labels if c.lower() in lowered), labels[-1]
        )
    try:
        confidence = max(0.0, min(1.0, float(value.get("confidence", 0))))
    except (TypeError, ValueError, AttributeError):
        confidence = 0.0
    return {
        "label": label,
        "confidence": confidence,
        "json": {"label": label, "confidence": confidence},
    }


@handler("llm.extract")
async def extract(ctx: NodeContext) -> dict:
    import json

    text = as_text(ctx.inputs.get("text")).strip()
    if not text:
        raise FlowNodeError("Connect the text to extract from")
    schema = _schema(ctx.params.get("json_schema"))
    if schema is None:
        raise FlowNodeError("Describe the fields to extract with a JSON schema")
    extra = str(ctx.params.get("instructions") or "").strip()
    system = (
        "Extract the requested fields from the text. Never invent a value: for anything the text does not say, "
        "use null where the schema allows it, otherwise an empty string, 0 or false."
    )
    if extra:
        system = f"{system}\n{extra}"
    value = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = system,
        prompt = text,
        schema = schema,
        temperature = 0.0,
        max_tokens = 2048,
        seed = ctx.seed,
    )
    return {"json": value, "text": json.dumps(value, ensure_ascii = False, indent = 1)}


@handler("chat.input")
async def chat_input(ctx: NodeContext) -> dict:
    supplied = ctx.flow_params or {}
    message = as_text(supplied.get("message")).strip() if "message" in supplied else ""
    if not message:
        message = str(ctx.params.get("test_message") or "").strip()
    return {"message": message, "history": chat_history(supplied.get("history"))}


@handler("chat.memory")
async def chat_memory(ctx: NodeContext) -> dict:
    turns = chat_history(ctx.inputs.get("history"))[-2 * int(ctx.params.get("turns") or 6) :]
    transcript = "\n".join(
        f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content']}" for t in turns
    )
    return {"history": turns, "text": transcript}


@handler("chat.output")
async def chat_output(ctx: NodeContext) -> dict:
    reply = as_text(ctx.inputs.get("reply")).strip()
    if not reply:
        raise FlowNodeError("The reply is empty")
    return {"reply": reply}
