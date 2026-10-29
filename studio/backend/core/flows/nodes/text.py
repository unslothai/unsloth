# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text nodes: LLM generation (streamed, or structured with a JSON schema), vision, templates, JSON and lists."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

from core.flows import model_settings
from core.flows.model_settings import request_params
from core.flows.nodes._common import as_text, first_json, lookup, render, resident_name, template_vars
from core.flows.registry import FlowNodeError, NodeContext, handler


def _schema(value: Any) -> dict | None:
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except ValueError:
            raise FlowNodeError("The JSON schema is not valid JSON") from None
    if isinstance(value, dict) and value:
        return value
    return None


def _prompt(ctx: NodeContext, param: str, port: str) -> str:
    variables = template_vars(ctx)
    template = str(ctx.params.get(param) or "")
    incoming = as_text(ctx.inputs.get(port)) if port in ctx.inputs else ""
    if template.strip():
        prompt = render(template, variables)
        if incoming and "{{" + port not in template.replace(" ", ""):
            prompt = f"{incoming}\n\n{prompt}"
    else:
        prompt = incoming
    context = ctx.inputs.get("context")
    if context and "{{context" not in template.replace(" ", ""):
        joined = "\n\n".join(as_text(c) for c in context)
        prompt = f"{prompt}\n\nContext:\n{joined}" if prompt else joined
    return prompt.strip()


async def _model_and_load(ctx: NodeContext, kind: str) -> tuple[Any, Optional[dict]]:
    """The model to ask for and the load options its card wants. A card on the loaded model that has load
    settings names that model, so a different load reloads it."""
    choice = ctx.params.get("model")
    load = ctx.params.get(model_settings.LOAD)
    if not load:
        return choice, None
    repo = choice.get("repo") if isinstance(choice, dict) else None
    variant = choice.get("variant") if isinstance(choice, dict) else None
    if not repo:
        repo = await resident_name(ctx, kind)
        if not repo:
            return choice, None
        choice = {"repo": repo, "variant": None}
    return choice, model_settings.load_kwargs(load, is_gguf = model_settings.looks_gguf(repo, variant))


def _temperature(ctx: NodeContext) -> float:
    value = ctx.params.get("temperature")
    return float(0.8 if value is None else value)


@handler("llm.generate")
async def generate(ctx: NodeContext) -> dict:
    prompt = _prompt(ctx, "prompt", "prompt")
    if not prompt:
        raise FlowNodeError("The prompt is empty: type one or connect text to Prompt")
    system = render(str(ctx.params.get("system") or ""), template_vars(ctx))
    images = [ctx.inputs["image"]] if ctx.inputs.get("image") else None
    schema = _schema(ctx.params.get("json_schema"))
    model, load = await _model_and_load(ctx, "vision" if images else "chat")
    if schema is not None:
        value = await ctx.llm_json(
            model = model,
            system = system,
            prompt = prompt,
            schema = schema,
            temperature = _temperature(ctx),
            max_tokens = int(ctx.params.get("max_tokens") or 1024),
            seed = ctx.seed,
            images = images,
            sampling = request_params(ctx.params),
            load = load,
        )
        return {"text": json.dumps(value, ensure_ascii = False, indent = 1), "json": value}
    text = await ctx.llm_text(
        model = model,
        system = system,
        prompt = prompt,
        temperature = _temperature(ctx),
        max_tokens = int(ctx.params.get("max_tokens") or 1024),
        images = images,
        thinking = bool(ctx.params.get("thinking")),
        seed = ctx.seed,
        stream_port = "text",
        sampling = request_params(ctx.params),
        load = load,
    )
    outputs: dict[str, Any] = {"text": text}
    try:
        outputs["json"] = first_json(text)
    except FlowNodeError:
        pass
    return outputs


@handler("llm.vision")
async def vision(ctx: NodeContext) -> dict:
    question = _prompt(ctx, "question", "question") or "Describe this image."
    system = render(str(ctx.params.get("system") or ""), template_vars(ctx))
    schema = _schema(ctx.params.get("json_schema"))
    images = [ctx.inputs["image"]]
    model, load = await _model_and_load(ctx, "vision")
    if schema is not None:
        value = await ctx.llm_json(
            model = model,
            system = system,
            prompt = question,
            schema = schema,
            temperature = _temperature(ctx),
            max_tokens = int(ctx.params.get("max_tokens") or 512),
            seed = ctx.seed,
            images = images,
            sampling = request_params(ctx.params),
            load = load,
        )
        return {"text": json.dumps(value, ensure_ascii = False), "json": value}
    text = await ctx.llm_text(
        model = model,
        system = system,
        prompt = question,
        temperature = _temperature(ctx),
        max_tokens = int(ctx.params.get("max_tokens") or 512),
        images = images,
        thinking = bool(ctx.params.get("thinking")),
        seed = ctx.seed,
        sampling = request_params(ctx.params),
        load = load,
    )
    outputs: dict[str, Any] = {"text": text}
    try:
        outputs["json"] = first_json(text)
    except FlowNodeError:
        pass
    return outputs


def _values_vars(ctx: NodeContext) -> dict[str, Any]:
    variables = dict(ctx.flow_params or {})
    values = ctx.inputs.get("values") or []
    for value in values:
        if isinstance(value, dict) and value.get("kind") not in ("image", "audio", "video"):
            variables.update({k: v for k, v in value.items() if isinstance(k, str)})
    variables["values"] = values[0] if len(values) == 1 else values
    for i, value in enumerate(values, 1):
        variables[f"v{i}"] = value
    return variables


@handler("text.template")
async def template(ctx: NodeContext) -> dict:
    return {"text": render(str(ctx.params.get("template") or ""), _values_vars(ctx))}


@handler("json.get")
async def json_get(ctx: NodeContext) -> dict:
    value = ctx.inputs.get("json")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = first_json(value)
    found = lookup(value, str(ctx.params.get("path") or ""))
    if found is None:
        raise FlowNodeError(f"Nothing at {ctx.params.get('path')!r}")
    return {"value": found}


def _fill(template: Any, variables: dict[str, Any]) -> Any:
    if isinstance(template, str):
        stripped = template.strip()
        if stripped.startswith("{{") and stripped.endswith("}}") and stripped.count("{{") == 1:
            name = stripped[2:-2].strip()
            head, _, rest = name.partition(".")
            if head in variables:
                return lookup(variables[head], rest) if rest else variables[head]
        return render(template, variables)
    if isinstance(template, list):
        return [_fill(v, variables) for v in template]
    if isinstance(template, dict):
        return {k: _fill(v, variables) for k, v in template.items()}
    return template


@handler("json.build")
async def json_build(ctx: NodeContext) -> dict:
    template = ctx.params.get("template")
    if isinstance(template, str):
        try:
            template = json.loads(template)
        except ValueError:
            raise FlowNodeError("The object is not valid JSON") from None
    return {"json": _fill(template if template is not None else {}, _values_vars(ctx))}


def _separator(value: Any) -> str:
    text = "\n" if value is None else str(value)
    return text.replace("\\n", "\n").replace("\\t", "\t")


MAX_CHUNKS = 10_000
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def chunk_text(text: str, size: int, overlap: int, unit: str) -> list[str]:
    """Overlapping chunks of ``size`` characters, words or sentences; ``overlap`` repeats that many units."""
    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size - 1))
    if unit == "characters":
        units, joiner = list(text), ""
    elif unit == "sentences":
        units, joiner = [u.strip() for u in _SENTENCE.split(text) if u.strip()], " "
    else:
        units, joiner = text.split(), " "
    step = size - overlap
    chunks: list[str] = []
    for start in range(0, len(units), step):
        piece = joiner.join(units[start : start + size]).strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(units):
            break
        if len(chunks) >= MAX_CHUNKS:
            raise FlowNodeError(f"More than {MAX_CHUNKS} chunks; raise the chunk size")
    return chunks


@handler("text.split")
async def split(ctx: NodeContext) -> dict:
    text = as_text(ctx.inputs.get("text"))
    if ctx.params.get("mode") == "chunks":
        return {
            "items": chunk_text(
                text,
                ctx.params.get("chunk_size") or 1000,
                ctx.params.get("overlap") or 0,
                str(ctx.params.get("unit") or "characters"),
            )
        }
    sep = _separator(ctx.params.get("separator"))
    parts = text.split(sep) if sep else list(text)
    if ctx.params.get("trim", True):
        parts = [p.strip() for p in parts if p.strip()]
    return {"items": parts}


@handler("list.join")
async def join(ctx: NodeContext) -> dict:
    items = ctx.inputs.get("items") or []
    return {"text": _separator(ctx.params.get("separator")).join(as_text(i) for i in items)}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


@handler("text.embed_rank")
async def embed_rank(ctx: NodeContext) -> dict:
    query = as_text(ctx.inputs.get("query"))
    items = [as_text(i) for i in ctx.inputs.get("items") or []]
    if not items:
        return {"ranked": [], "best": ""}
    answer = await ctx.call(
        "POST", "/v1/embeddings", json = {"model": "default", "input": [query, *items]}
    )
    rows = answer.get("data") if isinstance(answer, dict) else None
    if not isinstance(rows, list) or len(rows) != len(items) + 1:
        raise FlowNodeError("The embedding model returned no vectors")
    vectors = [r.get("embedding") for r in sorted(rows, key = lambda r: r.get("index", 0))]
    scored = sorted(
        zip(items, (_cosine(vectors[0], v) for v in vectors[1:])), key = lambda p: p[1], reverse = True
    )
    top = [text for text, _ in scored[: int(ctx.params.get("top_k") or 5)]]
    return {"ranked": top, "best": top[0]}
