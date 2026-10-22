# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""World Game nodes: generate a world with an LLM, paint its sprites and tiles, and answer NPC conversations."""

from __future__ import annotations

import copy
from typing import Any

from core.flows.expr import seed_from
from core.flows.registry import FlowNodeError, NodeContext, handler
from core.flows.story import slug
from core.flows.world import (
    BIOME_TILES,
    NPC_REPLY_SCHEMA,
    TILE_PROMPTS,
    WorldError,
    normalize_world,
    npc_prompt,
    validate_npc_reply,
    world_schema,
)

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
    npcs, items = int(p.get("npcs") or 4), int(p.get("items") or 4)
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


@handler("world.npc_reply")
async def world_npc_reply(ctx: NodeContext) -> dict:
    state = ctx.inputs.get("state")
    if not isinstance(state, dict):
        raise FlowNodeError("NPC Reply needs the conversation state from the world player.")
    try:
        system, prompt = npc_prompt(state, str(ctx.params.get("rules") or ""))
    except WorldError as exc:
        raise FlowNodeError(str(exc)) from exc
    raw = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = system,
        prompt = prompt,
        schema = NPC_REPLY_SCHEMA,
        temperature = 0.8,
        max_tokens = int(ctx.params.get("max_tokens") or 200) + 120,
        seed = seed_from(ctx.seed, state.get("npc_id"), len(state.get("history") or [])),
    )
    return {"reply": validate_npc_reply(raw, state)}


@handler("world.play")
async def world_play(ctx: NodeContext) -> dict:
    return {"world": _world_input(ctx)}
