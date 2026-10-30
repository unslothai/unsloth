# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Documents and knowledge bases, through Studio's RAG routes: the same parsers and the same per-account knowledge
bases chat searches."""

from __future__ import annotations

import asyncio
import re
from pathlib import PurePath
from typing import Any, Optional
from urllib.parse import quote

from core.flows import assets as asset_store
from core.flows.nodes._common import as_text, media_bytes, poll_job
from core.flows.registry import FlowNodeError, NodeContext, handler

RAG = "/api/rag"
POLL_S = 2.0


def _document(ctx: NodeContext) -> dict:
    value = ctx.inputs.get("file")
    if not (
        isinstance(value, dict)
        and value.get("kind") == "document"
        and isinstance(value.get("ref"), str)
    ):
        raise FlowNodeError("Connect a Document block (PDF, Word, Markdown, HTML or text)")
    return value


async def _upload(ctx: NodeContext) -> tuple[str, bytes, str]:
    value = _document(ctx)
    data, mime = await media_bytes(ctx, value)
    ext = asset_store.MIME_EXT.get(mime.split(";", 1)[0])
    if ext is None or mime.split(";", 1)[0] not in asset_store.DOCUMENT_MIMES:
        raise FlowNodeError("Load Document reads PDF, Word (.docx), Markdown, HTML and text files")
    stem = re.sub(r"[^\w .-]+", "_", PurePath(str(value.get("name") or "")).stem).strip(" ._")[:100]
    return f"{stem or 'document'}.{ext}", data, mime


@handler("input.file")
async def file_input(ctx: NodeContext) -> dict:
    value = ctx.params.get("asset")
    if not (
        isinstance(value, dict)
        and value.get("kind") == "document"
        and isinstance(value.get("ref"), str)
    ):
        raise FlowNodeError("Upload a document for this block")
    return {"file": value}


@handler("doc.load")
async def load_document(ctx: NodeContext) -> dict:
    name, data, mime = await _upload(ctx)
    answer = await ctx.call("POST", f"{RAG}/extract-text", files = {"file": (name, data, mime)})
    if not isinstance(answer, dict):
        raise FlowNodeError("Unsloth Studio could not read the document")
    return {"text": as_text(answer.get("text")), "truncated": bool(answer.get("truncated"))}


# Lookup-then-create is two requests, so parallel adds to a new knowledge base would each create one.
_KB_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}


async def _knowledge_base(ctx: NodeContext, create: bool) -> tuple[str, str]:
    name = str(ctx.params.get("knowledge_base") or "").strip()[:200].strip()
    if not name:
        raise FlowNodeError("Name the knowledge base")
    if not create:
        return await _find_or_create(ctx, name, create)
    key = (id(asyncio.get_running_loop()), name.lower())
    lock = _KB_LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        return await _find_or_create(ctx, name, create)


async def _find_or_create(ctx: NodeContext, name: str, create: bool) -> tuple[str, str]:
    listed = await ctx.call("GET", f"{RAG}/knowledge-bases")
    rows = listed.get("knowledgeBases") if isinstance(listed, dict) else None
    if isinstance(listed, dict) and listed.get("ragAvailable") is False:
        reason = listed.get("ragUnavailableReason") or "they cannot run on this install"
        raise FlowNodeError(f"Knowledge bases are not available: {reason}")
    for row in rows or []:
        if isinstance(row, dict) and str(row.get("name") or "").strip().lower() == name.lower():
            return str(row["id"]), str(row["name"])
    if not create:
        raise FlowNodeError(f"No knowledge base named {name!r}; add a document to it first")
    made = await ctx.call("POST", f"{RAG}/knowledge-bases", json = {"name": name})
    if not isinstance(made, dict) or not made.get("id"):
        raise FlowNodeError("Unsloth Studio did not create the knowledge base")
    return str(made["id"]), str(made.get("name") or name)


@handler("kb.add")
async def add_to_knowledge_base(ctx: NodeContext) -> dict:
    kb_id, kb_name = await _knowledge_base(ctx, create = True)
    name, data, mime = await _upload(ctx)
    started = await ctx.call(
        "POST",
        f"{RAG}/knowledge-bases/{quote(kb_id, safe = '')}/documents",
        files = {"file": (name, data, mime)},
    )
    job_id = started.get("jobId") if isinstance(started, dict) else None
    if not job_id:
        raise FlowNodeError("Unsloth Studio did not start indexing the document")

    def parse(status: Any) -> tuple[str, Optional[dict], Optional[str]]:
        state = str(status.get("status") or "") if isinstance(status, dict) else ""
        shown = None
        if isinstance(status, dict) and isinstance(status.get("progress"), (int, float)):
            shown = {
                "fraction": float(status["progress"]),
                "phase": str(status.get("stage") or "indexing"),
            }
        if state == "completed":
            return "completed", shown, None
        if state in ("failed", "cancelled"):
            return state, shown, str(status.get("error") or f"Indexing {state}")
        return "running", shown, None

    status = await poll_job(
        ctx,
        status = lambda: ctx.call("GET", f"{RAG}/jobs/{quote(str(job_id), safe = '')}"),
        parse = parse,
        interval = POLL_S,
    )
    chunks = status.get("numChunks") if isinstance(status, dict) else None
    return {"knowledge_base": kb_name, "chunks": int(chunks or 0)}


@handler("kb.search")
async def search_knowledge_base(ctx: NodeContext) -> dict:
    query = as_text(ctx.inputs.get("query")).strip()
    if not query:
        raise FlowNodeError("The question is empty")
    kb_id, _ = await _knowledge_base(ctx, create = False)
    mode = str(ctx.params.get("mode") or "hybrid")
    answer = await ctx.call(
        "POST",
        f"{RAG}/search",
        json = {
            "query": query,
            "kb_id": kb_id,
            "top_k": max(1, min(50, int(ctx.params.get("top_k") or 5))),
            "mode": mode if mode in ("hybrid", "dense", "lexical") else "hybrid",
        },
    )
    results = [
        r
        for r in (answer.get("results") if isinstance(answer, dict) else None) or []
        if isinstance(r, dict)
    ]
    passages = [as_text(r.get("text")) for r in results]
    context = "\n\n".join(
        f"[{i}] {r.get('filename') or 'document'}"
        + (f", page {r['page']}" if r.get("page") else "")
        + f"\n{as_text(r.get('text'))}"
        for i, r in enumerate(results, 1)
    )
    return {"passages": passages, "results": results, "context": context}
