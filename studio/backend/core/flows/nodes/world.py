# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""World Game nodes: generate a world with an LLM, paint its sprites and tiles, and answer NPC conversations."""

from __future__ import annotations

import asyncio
import copy
import re
from collections import OrderedDict
from typing import Any, Callable, Optional

from core.flows import memory as mem
from core.flows.city import MAYOR_SCHEMA, CityError, mayor_prompt, validate_mayor
from core.flows.authoring import AuthoringError, constrained_call
from core.flows.expr import seed_from
from core.flows.nodes._common import JsonFieldStream
from core.flows.nodes.memory import _embed, _plan_schema, _retrieve, clean_plan, rate_importance
from core.flows.registry import FlowNodeError, NodeContext, handler
from core.flows.rivals import RIVAL_TASKS
from core.flows.scene_solver import (
    SCENE_SCHEMA,
    SCENE_SYSTEM,
    Grid,
    fallback_scene,
    solve,
    validate_scene,
)
from core.flows.story import slug
from core.flows.world import (
    BIOME_TILES,
    NPC_MEMORY_REPLY_SCHEMA,
    NPC_REPLY_SCHEMA,
    DIRECTOR_SCHEMA,
    SUMMARY_SCHEMA,
    TILE_PROMPTS,
    WorldError,
    REFLECT_SCHEMA,
    chat_prompt,
    chat_schema,
    clean_insights,
    clean_options,
    clean_summary,
    compact_prompt,
    director_prompt,
    director_schema,
    fallback_quest,
    has_memory,
    normalize_world,
    npc_prompt,
    options_blocks,
    options_prompt,
    options_schema,
    plan_prompt,
    reflect_prompt,
    validate_chat,
    validate_director,
    validate_npc_reply,
    talk_summary,
    world_schema,
)

RECALL_K = 8
# text -> (embedding model, vector); a vector is only reused for the model that made it.
_VECTORS: "OrderedDict[str, tuple[Optional[str], list[float]]]" = OrderedDict()
_VECTORS_MAX = 20000
REPLY_RECALL_BUDGET_S = 0.6

WORLD_SYSTEM = """You design a tiny cozy top-down RPG world.
Give every NPC a distinct personality, a one-line greeting and a `sprite_prompt` describing their look.
Items are small objects the player can pick up. Each quest is given by an NPC (`giver` = npc id) and usually
needs one item (`requires_item` = item id, or "" for a talk quest). Ids are short snake_case."""


def _world_input(ctx: NodeContext) -> dict:
    w = ctx.inputs.get("world")
    if not isinstance(w, dict):
        raise FlowNodeError("Connect a world.")
    try:
        return normalize_world(w, seed = int(w.get("seed") or 0))
    except (WorldError, ValueError) as exc:
        raise FlowNodeError(
            f"The connected world is not valid: {str(exc).splitlines()[0]}"
        ) from exc


@handler("world.generator")
async def world_generator(ctx: NodeContext) -> dict:
    p = ctx.params
    premise = ctx.inputs.get("premise") if isinstance(ctx.inputs.get("premise"), str) else ""
    premise = (premise or str(p.get("premise") or "")).strip()
    if not premise:
        raise FlowNodeError("Give the World Generator a premise.")
    npcs = int(4 if p.get("npcs") is None else p["npcs"])
    items = int(4 if p.get("items") is None else p["items"])
    ctx.progress(0.05, phase = "designing")
    raw = await ctx.llm_json(
        model = p.get("model"),
        system = WORLD_SYSTEM,
        prompt = f"Premise: {premise}\nBiome: {p.get('biome') or 'meadow'}\n{npcs} NPCs, {items} items, 1-2 quests, "
        f"2-4 named regions.\nVariation seed: {ctx.seed}",
        schema = world_schema(npcs, items),
        temperature = 0.9,
        max_tokens = 4000,
        seed = ctx.seed,
    )
    if not isinstance(raw, dict):
        raise FlowNodeError("The model did not return a world", retryable = True)
    raw = dict(raw)
    raw["player"] = {
        "name": raw.get("player_name") or "You",
        "sprite_prompt": raw.get("player_sprite_prompt") or None,
    }
    raw["seed"] = ctx.seed
    try:
        world = normalize_world(
            raw, seed = ctx.seed, biome = p.get("biome"), style = p.get("style") or None
        )
    except WorldError as exc:
        raise FlowNodeError(str(exc), retryable = True) from exc
    ctx.log(
        f"{world['title']}: {len(world['npcs'])} NPCs, {len(world['items'])} items, {len(world['quests'])} quests"
    )
    return {"world": world}


def _style(world: dict) -> str:
    return str(world.get("style") or "cute 16-bit pixel art")


@handler("world.art")
async def world_art(ctx: NodeContext) -> dict:
    p = ctx.params
    world = copy.deepcopy(_world_input(ctx))
    style = _style(world)
    tiles = (
        [k for k in BIOME_TILES[world["biome"]] if k not in world["tileset"]]
        if p.get("tiles", True)
        else []
    )
    actors: list[tuple[str, dict, str]] = []
    if p.get("sprites", True):
        player = world["player"]
        if not player.get("sprite"):
            actors.append(
                (
                    "player",
                    player,
                    player.get("sprite_prompt") or f"a young adventurer named {player['name']}",
                )
            )
        for n in world["npcs"]:
            if not n.get("sprite"):
                actors.append(("npc", n, n.get("sprite_prompt") or n["name"]))
        for it in world["items"]:
            if not it.get("sprite"):
                actors.append(("item", it, it.get("sprite_prompt") or it["name"]))
    total = max(1, len(tiles) + len(actors))
    done = 0
    for kind in tiles:
        ctx.check_cancelled()
        ctx.progress(done / total, phase = f"tile: {kind}", step = done, total = total)
        imgs = await ctx.generate_image(
            model = p.get("model"),
            prompt = f"seamless tileable top-down {TILE_PROMPTS[kind]} texture, {style}, flat even lighting, "
            "game tile, fills the whole frame, no objects, no border, no text",
            width = 512,
            height = 512,
            seed = seed_from(ctx.seed, "tile", kind),
        )
        if imgs:
            world["tileset"][kind] = imgs[0]
        done += 1
    for role, ent, desc in actors:
        ctx.check_cancelled()
        name = ent.get("name") or role
        ctx.progress(done / total, phase = f"sprite: {name}", step = done, total = total)
        if role == "item":
            prompt = (
                f"{desc}, single game item icon, {style}, centered, plain white background, no text"
            )
        else:
            prompt = (
                f"{desc}, full body chibi game character sprite, front view, standing, {style}, centered, "
                "plain white background, no text"
            )
        imgs = await ctx.generate_image(
            model = p.get("model"),
            prompt = prompt,
            width = 768,
            height = 768,
            seed = seed_from(ctx.seed, role, ent.get("id") or name),
        )
        if not imgs:
            done += 1
            continue
        sprite = imgs[0]
        try:
            sprite, _mask = await ctx.remove_background(
                sprite, "the item" if role == "item" else "the character", trim = True
            )
        except FlowNodeError as exc:
            ctx.log(f"Kept the background on {name}: {exc}", "warning")
        ent["sprite"] = sprite
        done += 1
    ctx.progress(1.0, phase = "done", step = total, total = total)
    return {"world": normalize_world(world, seed = world["seed"])}


@handler("world.spec")
async def world_spec(ctx: NodeContext) -> dict:
    base: Any = ctx.inputs.get("world")
    if not isinstance(base, dict):
        base = ctx.params.get("world")
    if not isinstance(base, dict):
        raise FlowNodeError("Write the world in the inspector or connect one.")
    base = dict(base)
    ts = ctx.inputs.get("tileset")
    if isinstance(ts, dict):
        base["tileset"] = {**(base.get("tileset") or {}), **ts}
    try:
        return {"world": normalize_world(base, seed = int(base.get("seed") or ctx.seed))}
    except (WorldError, ValueError) as exc:
        raise FlowNodeError(f"The world is not valid: {str(exc).splitlines()[0]}") from exc


@handler("world.npc")
async def world_npc(ctx: NodeContext) -> dict:
    p = ctx.params
    name = str(p.get("name") or "Villager").strip()
    npc = {
        "id": slug(name, "npc"),
        "name": name,
        "persona": str(p.get("persona") or ""),
        "greeting": str(p.get("greeting") or ""),
    }
    sprite = ctx.inputs.get("sprite")
    if isinstance(sprite, dict):
        npc["sprite"] = sprite
    return {"npc": npc}


async def _recall(
    ctx: NodeContext,
    records: Any,
    query: str,
    now: float,
    k: int = RECALL_K,
    budget: Optional[float] = None,
) -> list[dict]:
    try:
        stream = mem.as_stream(records)
    except ValueError:
        return []
    for r in stream["records"]:
        r.pop("embedding", None)
    if not stream["records"]:
        return []
    query_vec = None
    if ctx.params.get("relevance", "embeddings") == "embeddings":
        texts = [r["text"] for r in stream["records"]]
        missing = list(dict.fromkeys(t for t in texts if t not in _VECTORS))
        pending = asyncio.ensure_future(_embed(ctx, [query] + missing))

        def keep(task: "asyncio.Future") -> None:
            got = None if task.cancelled() or task.exception() else task.result()
            vectors, model = got or ([], None)
            for text, vec in zip(missing, vectors[1:]):
                _VECTORS[text] = (model, vec)
            while len(_VECTORS) > _VECTORS_MAX:
                _VECTORS.popitem(last = False)

        pending.add_done_callback(keep)
        try:
            got = await (
                asyncio.wait_for(asyncio.shield(pending), budget) if budget is not None else pending
            )
        except asyncio.TimeoutError:
            got = None
            ctx.log("Memories ranked by keywords this turn: the embedding model was slow to answer")
        if got and got[0]:
            query_vec, model = got[0][0], got[1]
            keep(pending)
            for r in stream["records"]:
                cached_model, vec = _VECTORS.get(r["text"], (None, None))
                if vec is not None and cached_model == model and len(vec) == len(query_vec):
                    _VECTORS.move_to_end(r["text"])
                    r["embedding"] = vec
                else:
                    _VECTORS.pop(r["text"], None)
    picked = _retrieve(stream, query, query_vec, now, ctx.params, k, touch = False)
    return [
        {"id": r["id"], "text": r["text"], **({"source": r["source"]} if r.get("source") else {})}
        for r in picked
    ]


def _hours(state: dict) -> float:
    clock = state.get("clock")
    return (
        float(clock) / 60.0
        if isinstance(clock, (int, float)) and not isinstance(clock, bool)
        else 0.0
    )


@handler("world.npc_reply")
async def world_npc_reply(ctx: NodeContext) -> dict:
    state = ctx.inputs.get("state")
    if not isinstance(state, dict):
        raise FlowNodeError("NPC Reply needs the conversation state from the world player.")
    if state.get("task") == "summarise":
        return await _summarise(ctx, state)
    if state.get("task") == "director":
        return await _direct(ctx, state)
    if state.get("task") == "mayor":
        return await _mayor(ctx, state)
    if state.get("task") in RIVAL_TASKS:
        return await RIVAL_TASKS[state["task"]](ctx, state)
    if state.get("task") == "describe_place":
        return {"reply": {"place": await describe_place(ctx, state)}}
    memories: Optional[list[dict]] = None
    if has_memory(state):
        npc = next(
            (
                n
                for n in (state.get("world") or {}).get("npcs") or []
                if n.get("id") == state.get("npc_id")
            ),
            {},
        )
        query = f"{state.get('player_message') or ''} {npc.get('name') or ''}"
        memories = await _recall(
            ctx, state["memory"], query, _hours(state), budget = REPLY_RECALL_BUDGET_S
        )
    try:
        system, prompt = npc_prompt(state, str(ctx.params.get("rules") or ""), memories)
    except WorldError as exc:
        raise FlowNodeError(str(exc)) from exc
    spoken = JsonFieldStream("text")

    def on_text(so_far: str) -> None:
        piece = spoken.feed(so_far)
        if piece:
            ctx.text_delta("reply", piece)

    raw = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = system,
        prompt = prompt,
        schema = NPC_REPLY_SCHEMA if memories is None else NPC_MEMORY_REPLY_SCHEMA,
        temperature = 0.8,
        max_tokens = int(ctx.params.get("max_tokens") or 200) + (120 if memories is None else 200),
        on_text = on_text,
    )
    reply = validate_npc_reply(raw, state)
    if memories is not None:
        reply["recalled"] = [r["id"] for r in memories]
    return {"reply": reply}


async def _summarise(ctx: NodeContext, state: dict) -> dict:
    try:
        system, prompt = compact_prompt(state)
    except WorldError as exc:
        raise FlowNodeError(str(exc)) from exc
    raw = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = system,
        prompt = prompt,
        schema = SUMMARY_SCHEMA,
        temperature = 0.3,
        max_tokens = 320,
    )
    try:
        return {"reply": {"summary": clean_summary(raw)}}
    except WorldError as exc:
        raise FlowNodeError(str(exc), retryable = True) from exc


async def _direct(ctx: NodeContext, state: dict) -> dict:
    try:
        system, prompt = director_prompt(state)
    except WorldError as exc:
        raise FlowNodeError(str(exc)) from exc
    out = await constrained_call(
        ctx,
        director_schema(state),
        prompt,
        lambda raw: validate_director(raw, state),
        repair_max = 1,
        fallback = lambda: fallback_quest(state),
        system = system,
        temperature = 0.9,
        max_tokens = 500,
        defaults = {k: "" for k in DIRECTOR_SCHEMA["properties"]},
    )
    return {"reply": {"patch": out.value}}


async def _with_spec(place: Callable[[dict], Any], spec: dict) -> dict:
    return {**await place(spec), "spec": spec}


async def describe_place(ctx: NodeContext, req: dict) -> dict:
    """Lay out the described place on the request's map (rows of ``#`` / ``.`` / ``R``, or an open field)."""
    text = re.sub(r"\s+", " ", str(req.get("text") or ctx.params.get("description") or "")).strip()[
        :600
    ]
    if not text:
        raise FlowNodeError("Describe the place first.")
    try:
        Grid.from_request(req)
    except (AuthoringError, TypeError, ValueError, KeyError) as exc:
        raise FlowNodeError(f"Unreadable map: {exc}") from exc
    f = req.get("focus") if isinstance(req.get("focus"), dict) else {}
    focus = (
        (int(f["x"]), int(f["y"]))
        if all(isinstance(f.get(k), (int, float)) for k in ("x", "y"))
        else None
    )
    taken = {str(i) for i in req.get("taken") or [] if isinstance(i, str)}
    seed = int(req["seed"]) if isinstance(req.get("seed"), int) else int(ctx.seed or 0)

    async def place(spec: dict) -> dict:
        # A crowded map makes the solver search hard: keep it off the event loop.
        return await asyncio.to_thread(
            solve, spec, Grid.from_request(req), focus = focus, seed = seed, taken_ids = taken
        )

    async def check(raw: Any) -> dict:
        spec = validate_scene(raw)
        result = await place(spec)
        want = sum(p["count"] for p in spec["props"])
        if len(result["props3d"]) * 2 < want:
            raise AuthoringError(
                f"only {len(result['props3d'])} of {want} props fit on the free ground; ask for fewer"
            )
        return {**result, "spec": spec}

    out = await constrained_call(
        ctx,
        SCENE_SCHEMA,
        f"Describe this place as a scene: {text}\n",
        check,
        fallback = lambda: _with_spec(place, fallback_scene(text)),
        system = SCENE_SYSTEM,
        temperature = 0.7,
        max_tokens = 900,
    )
    return {**out.value, "fallback": out.fallback, "attempts": out.attempts}


@handler("world.describe_place")
async def world_describe_place(ctx: NodeContext) -> dict:
    req = ctx.inputs.get("request")
    req = dict(req) if isinstance(req, dict) else {}
    w = ctx.inputs.get("world")
    if not req.get("rows") and isinstance(w, dict):
        req.setdefault("width", w.get("width"))
        req.setdefault("height", w.get("height"))
    return {"place": await describe_place(ctx, req)}


async def _mayor(ctx: NodeContext, state: dict) -> dict:
    """One call a month (or per advisor click); an ungrounded answer falls back to fixed text, never a retry."""
    try:
        system, prompt = mayor_prompt(state)
    except CityError as exc:
        raise FlowNodeError(str(exc)) from exc
    raw = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = system,
        prompt = prompt,
        schema = MAYOR_SCHEMA,
        temperature = 0.7,
        max_tokens = 600,
    )
    return {"reply": {"mayor": validate_mayor(raw, state)}}


def _sim_world(state: Any, who: str) -> dict:
    if not isinstance(state, dict) or not isinstance(state.get("world"), dict):
        raise FlowNodeError(f"{who} needs the village state from the world player.")
    world = state["world"]
    if not (world.get("simulation") or {}).get("places"):
        raise FlowNodeError(f"{who} needs a world with a simulation section.")
    return world


def _npc(world: dict, npc_id: Any) -> dict:
    npc = next((n for n in world.get("npcs") or [] if n.get("id") == npc_id), None)
    if npc is None:
        raise FlowNodeError(f"Unknown NPC {npc_id!r}")
    return npc


RATE_MAX = 32


async def _rate(ctx: NodeContext, state: dict) -> dict:
    texts = [str(t)[:300] for t in state.get("texts") or [] if isinstance(t, str) and t.strip()]
    return {"plan": {"ratings": await rate_importance(ctx, texts[:RATE_MAX]) if texts else []}}


async def _reflect(ctx: NodeContext, world: dict, npc: dict, state: dict) -> dict:
    """One call instead of the paper's question-then-insight rounds."""
    records = [
        r
        for r in state.get("memory") or []
        if isinstance(r, dict) and isinstance(r.get("text"), str) and isinstance(r.get("id"), str)
    ][-30:]
    if not records:
        return {"plan": {"insights": []}}
    system, prompt = reflect_prompt(world, npc, records)
    raw = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = system,
        prompt = prompt,
        schema = REFLECT_SCHEMA,
        temperature = 0.7,
        max_tokens = 400,
    )
    return {"plan": {"npc": npc["id"], "insights": clean_insights(raw, records)}}


@handler("world.npc_plan")
async def world_npc_plan(ctx: NodeContext) -> dict:
    state = ctx.inputs.get("state")
    world = _sim_world(state, "NPC Plan")
    if state.get("task") == "rate":
        return await _rate(ctx, state)
    npc = _npc(world, state.get("npc_id"))
    if state.get("task") == "reflect":
        return await _reflect(ctx, world, npc, state)
    places = world["simulation"]["places"]
    names = [p["name"] for p in places]
    day = max(0, int(_hours(state) // 24))
    memories = await _recall(
        ctx, state.get("memory"), f"{npc['name']}'s plans for today", _hours(state)
    )
    if world["simulation"].get("options") and state.get("options"):
        system, prompt = options_prompt(
            world, npc, memories, state.get("clock"), state.get("needs")
        )
        raw = await ctx.llm_json(
            model = ctx.params.get("model"),
            system = system,
            prompt = prompt,
            schema = options_schema(names, [i["name"] for i in world.get("items") or []]),
            temperature = 0.7,
            max_tokens = 700,
        )
        options = clean_options(raw, world, state.get("clock"))
        return {
            "plan": {
                "npc": npc["id"],
                "day": day,
                "plan": options_blocks(options, state.get("clock")),
                "options": options,
            }
        }
    system, prompt = plan_prompt(world, npc, memories, day)
    raw = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = system,
        prompt = prompt,
        schema = _plan_schema(names, 6),
        temperature = 0.7,
        max_tokens = 600,
    )
    by_name = {p["name"]: p["id"] for p in places}
    plan = []
    for e in clean_plan(raw, names, 6):
        hh, mm = (int(x) for x in e["start"].split(":"))
        plan.append(
            {
                "start": day * 1440 + hh * 60 + mm,
                "minutes": e["minutes"],
                "activity": e["activity"][:80],
                "place": by_name[e["place"]],
            }
        )
    return {"plan": {"npc": npc["id"], "day": day, "plan": plan}}


@handler("world.npc_chat")
async def world_npc_chat(ctx: NodeContext) -> dict:
    state = ctx.inputs.get("state")
    world = _sim_world(state, "NPC Chat")
    a, b = _npc(world, state.get("a")), _npc(world, state.get("b"))
    if a["id"] == b["id"]:
        raise FlowNodeError("An NPC cannot chat with themselves.")
    now = _hours(state)
    mem_a = await _recall(ctx, state.get("memory_a"), b["name"], now, 5)
    mem_b = await _recall(ctx, state.get("memory_b"), a["name"], now, 5)
    social = bool(world["simulation"].get("relationships"))
    system, prompt = chat_prompt(world, a, b, mem_a, mem_b, state, social)
    raw = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = system,
        prompt = prompt,
        schema = chat_schema(a, b, social),
        temperature = 0.8,
        max_tokens = 700,
    )
    try:
        chat = validate_chat(raw, a, b, mem_a, mem_b) if social else validate_chat(raw, a, b)
    except WorldError as exc:
        raise FlowNodeError(str(exc), retryable = True) from exc
    chat["a_recalled"] = [r["id"] for r in mem_a]
    chat["b_recalled"] = [r["id"] for r in mem_b]
    return {"chat": chat}


@handler("world.play")
async def world_play(ctx: NodeContext) -> dict:
    return {"world": _world_input(ctx)}
