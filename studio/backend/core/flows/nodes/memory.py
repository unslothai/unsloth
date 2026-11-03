# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Generative Agents memory blocks; scoring lives in ``core.flows.memory``."""

from __future__ import annotations

import time
from typing import Any, Optional

from core.flows import memory as mem
from core.flows.nodes._common import as_text
from core.flows.registry import FlowNodeError, NodeContext, handler

EMBED_MODEL = "default"

IMPORTANCE_PROMPT = (
    "On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely "
    "poignant (e.g., a break up, college acceptance), rate the likely poignancy of each of the following pieces "
    "of memory. Answer with one rating per memory, in order."
)
RATINGS_SCHEMA = {
    "type": "object",
    "properties": {
        "ratings": {"type": "array", "items": {"type": "integer", "minimum": 1, "maximum": 10}}
    },
    "required": ["ratings"],
    "additionalProperties": False,
}
QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
    "additionalProperties": False,
}
INSIGHTS_SCHEMA = {
    "type": "object",
    "properties": {
        "insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "cites": {"type": "array", "items": {"type": "integer"}},
                    "importance": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["text", "cites", "importance"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["insights"],
    "additionalProperties": False,
}
SUMMARY_QUERIES = (
    "{name}'s core characteristics",
    "{name}'s current daily occupation",
    "{name}'s feeling about their recent progress in life",
)


def _now(ctx: NodeContext) -> float:
    value = ctx.inputs.get("now")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return time.time() / 3600.0


def _stream(ctx: NodeContext) -> dict:
    try:
        return mem.as_stream(ctx.inputs.get("memory"))
    except ValueError as exc:
        raise FlowNodeError(f"{exc}; connect the memory output of another memory block") from None


async def embed(ctx: NodeContext, texts: list[str]) -> Optional[list[list[float]]]:
    """Vectors from Studio's local embedding model, or None (the blocks then rank by keywords)."""
    got = await _embed(ctx, texts)
    return None if got is None else got[0]


def _set_vectors(records: list[dict], vectors: Optional[list], model: Optional[str]) -> None:
    for record, vec in zip(records, vectors or []):
        record["embedding"] = vec
        record["embed_model"] = model


# routes/inference.py _STUDIO_EMBED_MAX_INPUTS; streams hold up to MAX_RECORDS (5000).
EMBED_BATCH = 2048


async def _embed(ctx: NodeContext, texts: list[str]) -> Optional[tuple[list, Optional[str]]]:
    if not texts:
        return [], None
    vectors: list = []
    model: Optional[str] = None
    for start in range(0, len(texts), EMBED_BATCH):
        got = await _embed_batch(ctx, texts[start : start + EMBED_BATCH])
        if got is None or (start and got[1] != model):
            return None
        vectors += got[0]
        model = got[1]
    return vectors, model


async def _embed_batch(ctx: NodeContext, texts: list[str]) -> Optional[tuple[list, Optional[str]]]:
    try:
        answer = await ctx.call(
            "POST", "/v1/embeddings", json = {"model": EMBED_MODEL, "input": texts}
        )
    except FlowNodeError as exc:
        ctx.log(f"No embedding model answered ({exc}); ranking by keywords instead", "warning")
        return None
    rows = answer.get("data") if isinstance(answer, dict) else None
    if not isinstance(rows, list) or len(rows) != len(texts):
        ctx.log("The embedding model returned no vectors; ranking by keywords instead", "warning")
        return None
    vectors = [
        mem.round_vector(r.get("embedding")) for r in sorted(rows, key = lambda r: r.get("index", 0))
    ]
    if any(v is None for v in vectors):
        return None
    model = answer.get("model")
    return vectors, model if isinstance(model, str) else None


async def _embed_missing(
    ctx: NodeContext, records: list[dict], queries: list[str]
) -> Optional[list]:
    """One embedding call for the queries plus any record still without a vector; fills records in place.

    Vectors from a different embedding model live in another space, so those records are embedded again."""
    missing = [r for r in records if not isinstance(r.get("embedding"), list)]
    got = await _embed(ctx, queries + [r["text"] for r in missing])
    if got is None:
        return None
    vectors, model = got
    _set_vectors(missing, vectors[len(queries) :], model)
    dim = len(vectors[0]) if vectors else 0
    stale = [r for r in records if r.get("embed_model") != model or len(r["embedding"]) != dim]
    if stale:
        again = await _embed(ctx, [r["text"] for r in stale])
        if again is None or again[1] != model:
            return None
        _set_vectors(stale, again[0], model)
    return vectors[: len(queries)]


RATE_BATCH = 32


async def rate_importance(ctx: NodeContext, texts: list[str]) -> list[int]:
    """The paper's poignancy prompt, up to RATE_BATCH texts per structured call so a bulk import fits a
    local model's context."""
    out: list[int] = []
    for start in range(0, len(texts), RATE_BATCH):
        out += await _rate_batch(ctx, texts[start : start + RATE_BATCH])
    return out


async def _rate_batch(ctx: NodeContext, texts: list[str]) -> list[int]:
    listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    raw = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = "You rate memories for an agent. Answer in JSON.",
        prompt = f"{IMPORTANCE_PROMPT}\n\nMemories:\n{listing}",
        schema = RATINGS_SCHEMA,
        temperature = 0.0,
        max_tokens = 16 + 8 * len(texts),
        seed = ctx.seed,
    )
    ratings = raw.get("ratings") if isinstance(raw, dict) else None
    ratings = ratings if isinstance(ratings, list) else []
    return [mem.clamp_importance(ratings[i]) if i < len(ratings) else 5 for i in range(len(texts))]


def _items(value: Any) -> list[dict]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    items = []
    for v in values:
        if isinstance(v, dict) and v.get("text"):
            items.append(v)
        else:
            text = as_text(v).strip()
            if text:
                items.append({"text": text})
    return items


@handler("memory.add")
async def add(ctx: NodeContext) -> dict:
    stream = _stream(ctx)
    items = _items(ctx.inputs.get("items"))
    if not items:
        return {"memory": stream, "added": []}
    now = _now(ctx)
    kind = str(ctx.params.get("kind") or "observation")
    unrated = [i for i, it in enumerate(items) if it.get("importance") is None]
    if unrated and ctx.params.get("importance") == "llm":
        rated = await rate_importance(ctx, [str(items[i]["text"]) for i in unrated])
        for i, r in zip(unrated, rated):
            items[i] = {**items[i], "importance": r}
    fixed = ctx.params.get("fixed_importance") or 5
    added = [
        mem.new_record(
            stream,
            str(it["text"]),
            str(it.get("kind") or kind),
            it.get("importance") if it.get("importance") is not None else fixed,
            now,
            cites = it.get("cites") if isinstance(it.get("cites"), list) else None,
        )
        for it in items
    ]
    if ctx.params.get("embed", True):
        got = await _embed(ctx, [r["text"] for r in added])
        if got is not None:
            _set_vectors(added, *got)
    stream["records"].extend(added)
    stream["since_reflection"] += sum(r["importance"] for r in added)
    mem.cap(stream, int(ctx.params.get("max_records") or 500))
    return {"memory": stream, "added": added}


def _weights(params: dict) -> dict:
    return {
        "recency": float(params.get("w_recency", mem.DEFAULT_WEIGHTS["recency"])),
        "importance": float(params.get("w_importance", mem.DEFAULT_WEIGHTS["importance"])),
        "relevance": float(params.get("w_relevance", mem.DEFAULT_WEIGHTS["relevance"])),
    }


def _retrieve(
    stream: dict,
    query: str,
    query_vec: Optional[list],
    now: float,
    params: dict,
    k: int,
    touch: bool,
) -> list[dict]:
    records = stream["records"]
    if not records:
        return []
    scores = mem.score(
        records,
        now = now,
        query = query,
        query_vec = query_vec,
        weights = _weights(params),
        decay = float(params.get("decay") or mem.DEFAULT_DECAY),
        recency_mode = str(params.get("recency_mode") or "hours"),
    )
    picked = []
    for i in mem.top(records, scores, k):
        if touch:
            records[i]["t_access"] = now
        row = {
            key: val for key, val in records[i].items() if key not in ("embedding", "embed_model")
        }
        picked.append({**row, **{f"score_{n}": round(v, 4) for n, v in scores[i].items()}})
    return picked


@handler("memory.retrieve")
async def retrieve(ctx: NodeContext) -> dict:
    stream = _stream(ctx)
    query = as_text(ctx.inputs.get("query")).strip()
    now = _now(ctx)
    query_vec = None
    if stream["records"] and query and ctx.params.get("relevance", "embeddings") == "embeddings":
        vectors = await _embed_missing(ctx, stream["records"], [query])
        query_vec = vectors[0] if vectors else None
    picked = _retrieve(
        stream,
        query,
        query_vec,
        now,
        ctx.params,
        int(ctx.params.get("top_k") or 8),
        bool(ctx.params.get("touch", True)),
    )
    return {"memory": stream, "records": picked, "context": mem.context_text(picked)}


def _who(ctx: NodeContext) -> str:
    return (
        as_text(ctx.inputs.get("name")).strip()
        or str(ctx.params.get("name") or "").strip()
        or "the agent"
    )


@handler("memory.reflect")
async def reflect(ctx: NodeContext) -> dict:
    stream = _stream(ctx)
    threshold = float(ctx.params.get("threshold") or 150)
    if not stream["records"] or (
        not ctx.params.get("force") and stream["since_reflection"] < threshold
    ):
        return {"memory": stream, "insights": [], "reflected": False}
    now, who = _now(ctx), _who(ctx)
    recent = sorted(stream["records"], key = lambda r: float(r.get("t_created") or 0))[
        -int(ctx.params.get("recent") or 100) :
    ]
    raw = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = f"You help {who} reflect on their memories. Answer in JSON.",
        prompt = mem.context_text(recent)
        + "\n\nGiven only the information above, what are the 3 most salient high-level questions we can answer "
        "about the subjects in the statements?",
        schema = QUESTIONS_SCHEMA,
        temperature = 0.7,
        max_tokens = 300,
        seed = ctx.seed,
    )
    asked = raw.get("questions") if isinstance(raw, dict) else None
    asked = [asked] if isinstance(asked, str) else asked if isinstance(asked, list) else []
    questions = [str(q).strip() for q in asked if str(q).strip()][:3]
    if not questions:
        stream["since_reflection"] = 0.0
        return {"memory": stream, "insights": [], "reflected": False}
    vectors = await _embed_missing(ctx, stream["records"], questions)
    per_question = int(ctx.params.get("max_insights") or 5)
    new: list[dict] = []
    for qi, question in enumerate(questions):
        evidence = _retrieve(
            stream, question, vectors[qi] if vectors else None, now, ctx.params, 8, True
        )
        if not evidence:
            continue
        raw = await ctx.llm_json(
            model = ctx.params.get("model"),
            system = f"You help {who} reflect on their memories. Answer in JSON.",
            prompt = f"Statements about {who}:\n{mem.context_text(evidence)}\n\n"
            f"What {per_question} high-level insights can you infer from the above statements? For each, list the "
            "numbers of the statements it rests on, and rate its importance from 1 to 10.",
            schema = INSIGHTS_SCHEMA,
            temperature = 0.7,
            max_tokens = 120 * per_question,
            seed = (ctx.seed + qi) % (2**32),
        )
        found = raw.get("insights") if isinstance(raw, dict) else None
        found = [found] if isinstance(found, dict) else found if isinstance(found, list) else []
        for insight in found[:per_question]:
            if not isinstance(insight, dict) or not str(insight.get("text") or "").strip():
                continue
            cites = [
                evidence[n - 1]["id"]
                for n in (insight.get("cites") if isinstance(insight.get("cites"), list) else [])
                if isinstance(n, int) and 1 <= n <= len(evidence)
            ]
            new.append(
                mem.new_record(
                    stream,
                    str(insight["text"]).strip(),
                    "reflection",
                    insight.get("importance"),
                    now,
                    cites = cites,
                )
            )
    if new:
        got = await _embed(ctx, [r["text"] for r in new])
        if got is not None:
            _set_vectors(new, *got)
        stream["records"].extend(new)
    stream["since_reflection"] = 0.0
    return {"memory": stream, "insights": [r["text"] for r in new], "reflected": bool(new)}


@handler("agent.summary")
async def summary(ctx: NodeContext) -> dict:
    stream = _stream(ctx)
    who, now = _who(ctx), _now(ctx)
    queries = [q.format(name = who) for q in SUMMARY_QUERIES]
    vectors = await _embed_missing(ctx, stream["records"], queries) if stream["records"] else None
    seen: dict[str, dict] = {}
    for qi, query in enumerate(queries):
        for row in _retrieve(
            stream, query, vectors[qi] if vectors else None, now, ctx.params, 8, False
        ):
            seen.setdefault(row["id"], row)
    persona = as_text(ctx.inputs.get("persona")).strip()
    if not seen and not persona:
        return {"summary": ""}
    statements = mem.context_text(seen.values())
    text = await ctx.llm_text(
        model = ctx.params.get("model"),
        system = "You write short, factual character summaries in plain prose.",
        prompt = (f"Known traits: {persona}\n\n" if persona else "")
        + f"Statements about {who}:\n{statements}\n\n"
        f"In 3 to 5 sentences, describe {who}'s core characteristics, current daily occupation, and how they feel "
        "about their recent progress in life, using only the statements above.",
        temperature = 0.5,
        max_tokens = 300,
        seed = ctx.seed,
        stream_port = "summary",
    )
    return {"summary": text.strip()}


def _plan_schema(places: list[str], chunks: int) -> dict:
    place: dict[str, Any] = {"type": "string"}
    if places:
        place["enum"] = places
    return {
        "type": "object",
        "properties": {
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "pattern": "^([01][0-9]|2[0-3]):[0-5][0-9]$"},
                        "minutes": {"type": "integer", "minimum": 5, "maximum": 720},
                        "activity": {"type": "string"},
                        "place": place,
                    },
                    "required": ["start", "minutes", "activity", "place"],
                    "additionalProperties": False,
                },
                "maxItems": chunks,
            }
        },
        "required": ["plan"],
        "additionalProperties": False,
    }


def clean_plan(raw: Any, places: list[str], chunks: int) -> list[dict]:
    """Keep well-formed entries in time order; a place outside the known list is dropped, not guessed."""
    plan = []
    entries = raw.get("plan") if isinstance(raw, dict) else None
    entries = (
        [entries] if isinstance(entries, dict) else entries if isinstance(entries, list) else []
    )
    for entry in entries:
        if len(plan) >= chunks:
            break
        if not isinstance(entry, dict) or not str(entry.get("activity") or "").strip():
            continue
        start = str(entry.get("start") or "")
        try:
            hh, mm = (int(x) for x in start.split(":"))
        except ValueError:
            continue
        if not (0 <= hh < 24 and 0 <= mm < 60):
            continue
        place = str(entry.get("place") or "")
        if places and place not in places:
            continue
        try:
            minutes = max(5, min(720, int(entry.get("minutes") or 60)))
        except (TypeError, ValueError):
            continue
        plan.append(
            {
                "start": f"{hh:02d}:{mm:02d}",
                "minutes": minutes,
                "activity": str(entry["activity"]).strip(),
                "place": place,
            }
        )
    return sorted(plan, key = lambda e: e["start"])


@handler("agent.plan")
async def plan(ctx: NodeContext) -> dict:
    who = _who(ctx)
    places = [
        p
        for p in (
            str(x).strip() for x in ctx.inputs.get("places") or ctx.params.get("places") or []
        )
        if p
    ]
    chunks = int(ctx.params.get("chunks") or 6)
    start_hour = ctx.params.get("start_hour")
    start = 8 if start_hour is None else int(start_hour)
    summary_text = as_text(ctx.inputs.get("summary")).strip()
    recent = as_text(ctx.inputs.get("context")).strip()
    raw = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = f"You plan {who}'s day. Answer in JSON.",
        prompt = (f"{summary_text}\n\n" if summary_text else "")
        + (f"Recent memories:\n{recent}\n\n" if recent else "")
        + f"Today is {ctx.params.get('day') or 'today'}. Make {who}'s plan for the day from {start:02d}:00 in "
        f"{chunks} or fewer broad strokes, each with a start time (HH:MM), length in minutes, activity and place"
        + (f" (one of: {', '.join(places)})." if places else "."),
        schema = _plan_schema(places, chunks),
        temperature = 0.7,
        max_tokens = 80 * chunks + 60,
        seed = ctx.seed,
    )
    entries = clean_plan(raw, places, chunks)
    text = "\n".join(
        f"{e['start']} ({e['minutes']} min) {e['activity']}"
        + (f" at {e['place']}" if e["place"] else "")
        for e in entries
    )
    return {"plan": entries, "text": text}
