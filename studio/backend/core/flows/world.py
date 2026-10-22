# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""World specs: pick the world a flow produced, repair LLM-written worlds and validate NPC replies.

The map itself (terrain, paths, where things stand) is generated in the player from `seed` and `biome`
(`studio/frontend/src/features/flows/player/world/procgen.ts`), so a world spec stays small and shareable."""

from __future__ import annotations

import re
from typing import Any, Optional

from core.flows.expr import Prng, seed_from
from core.flows.schema import WorldSpec
from core.flows.story import _unique, input_values, most_downstream, node_value, slug

WORLD_SOURCE_TYPES = ("world.play", "world.spec", "world.art", "world.generator")
WORLD_ENTRY_TYPES = {"world.npc_reply": "npc_reply"}
BIOMES = ("meadow", "forest", "desert", "snow", "town", "dungeon", "beach")
# Tile kinds each biome's map uses (procgen.ts draws the same kinds).
BIOME_TILES = {
    "meadow": ("grass", "path", "water", "flowers", "tree", "rock"),
    "forest": ("grass", "path", "water", "tree", "rock", "flowers"),
    "desert": ("sand", "path", "water", "rock"),
    "snow": ("snow", "path", "water", "tree", "rock"),
    "town": ("grass", "path", "floor", "wall", "water", "tree"),
    "dungeon": ("floor", "wall", "path", "water", "rock"),
    "beach": ("sand", "water", "grass", "path", "rock", "tree"),
}
TILE_PROMPTS = {
    "grass": "lush green grass",
    "path": "packed dirt path with small pebbles",
    "water": "calm blue water with soft ripples",
    "sand": "warm golden sand",
    "snow": "fresh white snow",
    "floor": "wooden floor boards",
    "wall": "grey stone brick wall",
    "flowers": "grass with small colourful wildflowers",
    "tree": "dense tree canopy seen from above",
    "rock": "mossy grey boulders",
}
MAX_HISTORY = 24


class WorldError(ValueError):
    pass


def _clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    return max(lo, min(hi, int(v)))


def normalize_world(
    raw: Any,
    *,
    seed: int = 0,
    biome: Optional[str] = None,
    style: Optional[str] = None,
) -> dict:
    """Coerce a loosely shaped world (LLM output or hand-written JSON) into a valid WorldSpec dict."""
    if not isinstance(raw, dict):
        raise WorldError("the world is not an object")
    rng = Prng(seed_from("world", seed))
    width = _clamp_int(raw.get("width"), 16, 256, 48)
    height = _clamp_int(raw.get("height"), 16, 256, 36)
    tile = _clamp_int(raw.get("tile_size"), 8, 128, 32)
    b = biome or raw.get("biome")
    b = b if b in BIOMES else "meadow"

    taken: set[str] = set()

    def ids(items: Any, fallback: str) -> list[dict]:
        out = []
        for it in items or []:
            if isinstance(it, dict) and str(it.get("name") or "").strip():
                it = dict(it)
                it["id"] = _unique(slug(it.get("id") or it["name"], fallback), taken)
                out.append(it)
        return out

    npcs_raw = ids(raw.get("npcs"), "npc")[:32]
    items_raw = ids(raw.get("items"), "item")[:64]
    name_to_id = {}
    for it in npcs_raw + items_raw:
        name_to_id[str(it["name"]).lower()] = it["id"]
        name_to_id[str(it.get("id")).lower()] = it["id"]

    def ref(v: Any, pool: set[str]) -> Optional[str]:
        key = str(v or "").strip().lower()
        hit = name_to_id.get(key) or name_to_id.get(slug(key))
        return hit if hit in pool else None

    def pos(it: dict) -> dict:
        x, y = it.get("x"), it.get("y")
        ok = isinstance(x, int) and isinstance(y, int) and 0 <= x < width and 0 <= y < height
        return {"x": x, "y": y} if ok else {}

    npc_ids = {n["id"] for n in npcs_raw}
    item_ids = {i["id"] for i in items_raw}
    npcs = [
        {
            "id": n["id"],
            "name": str(n["name"])[:60],
            "persona": str(n.get("persona") or "")[:1500],
            "greeting": str(n.get("greeting") or f"Hello, I'm {n['name']}.")[:300],
            "sprite": n.get("sprite") if isinstance(n.get("sprite"), dict) else None,
            "sprite_prompt": str(n.get("sprite_prompt") or n.get("look") or "")[:400] or None,
            "wander": n.get("wander", True) is not False,
            **pos(n),
        }
        for n in npcs_raw
    ]
    items = [
        {
            "id": i["id"],
            "name": str(i["name"])[:60],
            "description": str(i.get("description") or "")[:500],
            "sprite": i.get("sprite") if isinstance(i.get("sprite"), dict) else None,
            "sprite_prompt": str(i.get("sprite_prompt") or "")[:400] or None,
            **pos(i),
        }
        for i in items_raw
    ]
    quests = []
    for q in raw.get("quests") or []:
        if not isinstance(q, dict) or not str(q.get("title") or "").strip():
            continue
        giver = ref(q.get("giver"), npc_ids)
        if giver is None:
            if not npcs:
                continue
            giver = npcs[rng.randint(0, len(npcs) - 1)]["id"]
        quests.append(
            {
                "id": _unique(slug(q.get("id") or q["title"], "quest"), taken),
                "title": str(q["title"])[:120],
                "description": str(q.get("description") or "")[:800],
                "giver": giver,
                "requires_item": ref(q.get("requires_item"), item_ids),
                "reward_text": str(q.get("reward_text") or "Thank you!")[:400],
                "set": q.get("set") if isinstance(q.get("set"), dict) else {},
            }
        )
    regions = []
    raw_regions = [
        r
        for r in raw.get("regions") or []
        if isinstance(r, dict) and str(r.get("name") or "").strip()
    ][:9]
    cols = 3
    for i, r in enumerate(raw_regions):
        cw, ch = width // cols, height // cols
        gx, gy = i % cols, i // cols
        x, y, w, h = (r.get(k) for k in ("x", "y", "w", "h"))
        if not all(isinstance(v, int) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            x, y, w, h = gx * cw + 1, gy * ch + 1, max(cw - 2, 3), max(ch - 2, 3)
        regions.append(
            {
                "id": _unique(slug(r.get("id") or r["name"], "region"), taken),
                "name": str(r["name"])[:60],
                "x": max(0, min(width - 1, x)),
                "y": max(0, min(height - 1, y)),
                "w": max(1, min(width, w)),
                "h": max(1, min(height, h)),
                "on_enter_text": str(r.get("on_enter_text") or "")[:300],
            }
        )
    player = raw.get("player") if isinstance(raw.get("player"), dict) else {}
    vars_ = raw.get("vars") if isinstance(raw.get("vars"), dict) else {}
    tileset = {
        k: v
        for k, v in (raw.get("tileset") or {}).items()
        if isinstance(v, dict) and k in TILE_PROMPTS
    }
    world = {
        "title": str(raw.get("title") or "A Small World")[:200],
        "seed": int(raw["seed"])
        if isinstance(raw.get("seed"), int) and raw["seed"] >= 0
        else seed & 0xFFFFFFFF,
        "width": width,
        "height": height,
        "tile_size": tile,
        "biome": b,
        "tileset": tileset,
        "player": {
            "name": str(player.get("name") or raw.get("player_name") or "You")[:60],
            **({"sprite": player["sprite"]} if isinstance(player.get("sprite"), dict) else {}),
            **(
                {"sprite_prompt": str(player["sprite_prompt"])[:400]}
                if player.get("sprite_prompt")
                else {}
            ),
        },
        "npcs": npcs,
        "items": items,
        "quests": quests,
        "regions": regions,
        "vars": {slug(k): v for k, v in vars_.items() if isinstance(v, (int, float, str, bool))},
        "intro_text": str(raw.get("intro_text") or "")[:1200],
        "style": style or raw.get("style") or None,
        "entry_points": raw.get("entry_points")
        if isinstance(raw.get("entry_points"), dict)
        else {},
    }
    return WorldSpec.model_validate(world).model_dump(exclude_none = True)


def world_schema(npcs: int, items: int) -> dict:
    npc = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "persona": {"type": "string"},
            "greeting": {"type": "string"},
            "sprite_prompt": {"type": "string"},
        },
        "required": ["id", "name", "persona", "greeting", "sprite_prompt"],
    }
    item = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "sprite_prompt": {"type": "string"},
        },
        "required": ["id", "name", "description", "sprite_prompt"],
    }
    quest = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "giver": {"type": "string"},
            "requires_item": {"type": "string"},
            "reward_text": {"type": "string"},
        },
        "required": ["id", "title", "description", "giver", "requires_item", "reward_text"],
    }
    region = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "on_enter_text": {"type": "string"}},
        "required": ["name", "on_enter_text"],
    }
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "intro_text": {"type": "string"},
            "player_name": {"type": "string"},
            "player_sprite_prompt": {"type": "string"},
            "npcs": {"type": "array", "items": npc, "minItems": 1, "maxItems": max(1, npcs)},
            "items": {"type": "array", "items": item, "maxItems": max(0, items)},
            "quests": {"type": "array", "items": quest, "minItems": 1, "maxItems": 3},
            "regions": {"type": "array", "items": region, "maxItems": 4},
        },
        "required": ["title", "intro_text", "player_name", "npcs", "items", "quests", "regions"],
    }


NPC_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "give_item": {"type": "string"},
        "take_item": {"type": "string"},
        "complete_quest": {"type": "string"},
        "set": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {"var": {"type": "string"}, "value": {"type": "number"}},
                "required": ["var", "value"],
            },
        },
    },
    "required": ["text"],
}


def npc_context(state: dict) -> tuple[dict, dict]:
    world = state.get("world") if isinstance(state.get("world"), dict) else None
    if world is None:
        raise WorldError("npc_reply needs the world in its state")
    npc = next((n for n in world.get("npcs") or [] if n.get("id") == state.get("npc_id")), None)
    if npc is None:
        raise WorldError(f"unknown NPC {state.get('npc_id')!r}")
    return world, npc


def validate_npc_reply(raw: Any, state: dict) -> dict:
    """Keep only effects the world allows: known items, the player's inventory, this NPC's quests, declared vars."""
    world, npc = npc_context(state)
    if isinstance(raw, str):
        raw = {"text": raw}
    if not isinstance(raw, dict):
        raise WorldError("the model did not return a reply")
    inventory = [i for i in state.get("inventory") or [] if isinstance(i, str)]
    item_ids = {i["id"] for i in world.get("items") or []}
    done = set(state.get("completed_quests") or [])
    text = re.sub(r"\s+", " ", str(raw.get("text") or "")).strip()[:1200] or "..."
    out: dict[str, Any] = {"text": text}
    give = str(raw.get("give_item") or "").strip()
    if give in item_ids and give not in inventory:
        out["give_item"] = give
    take = str(raw.get("take_item") or "").strip()
    if take in inventory:
        out["take_item"] = take
    qid = str(raw.get("complete_quest") or "").strip()
    quest = next((q for q in world.get("quests") or [] if q["id"] == qid), None)
    if quest and quest["giver"] == npc["id"] and qid not in done:
        need = quest.get("requires_item")
        if not need or need in inventory:
            out["complete_quest"] = qid
            if need and "take_item" not in out:
                out["take_item"] = need
    sets = {}
    declared = world.get("vars") or {}
    for s in raw.get("set") or []:
        if isinstance(s, dict) and s.get("var") in declared:
            v = s.get("value")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sets[s["var"]] = v
    if sets:
        out["set"] = sets
    return out


def npc_prompt(state: dict, rules: str) -> tuple[str, str]:
    world, npc = npc_context(state)
    items = {i["id"]: i for i in world.get("items") or []}
    inv = [items[i]["name"] + f" (id {i})" for i in state.get("inventory") or [] if i in items]
    quests = [q for q in world.get("quests") or [] if q["giver"] == npc["id"]]
    done = set(state.get("completed_quests") or [])
    q_lines = [
        f"- {q['title']} (id {q['id']}): {q['description']}"
        + (f" Needs item id {q['requires_item']}." if q.get("requires_item") else "")
        + (" ALREADY COMPLETED." if q["id"] in done else "")
        for q in quests
    ]
    system = (
        f"You are {npc['name']}, a character in the game world \"{world.get('title')}\".\n"
        f"Persona: {npc.get('persona') or 'a friendly local'}\n"
        f"World: {world.get('intro_text') or ''}\n"
        f"{rules}\n"
        "Reply as JSON. `text` is what you say. Only set `complete_quest` to one of your quest ids when the player has "
        "earned it (they carry the needed item). Use `give_item` only for an item id from the world list that fits the "
        "story; use empty strings when nothing happens."
    )
    history = (state.get("history") or [])[-MAX_HISTORY:]
    convo = "\n".join(
        f"{'Player' if h.get('role') == 'player' else npc['name']}: {str(h.get('text') or '')[:400]}"
        for h in history
        if isinstance(h, dict)
    )
    item_list = ", ".join("{} (id {})".format(i["name"], i["id"]) for i in items.values())
    quest_list = "\n".join(q_lines)
    prompt = (
        f"Your quests:\n{quest_list or '- none'}\n"
        f"World items: {item_list or 'none'}\n"
        f"Player ({(world.get('player') or {}).get('name', 'Player')}) carries: {', '.join(inv) or 'nothing'}\n"
        f"Variables: {state.get('vars') or {}}\n"
        f"Conversation so far:\n{convo or '(just met)'}\n"
        f"Player: {str(state.get('player_message') or '')[:600]}\n"
        f"{npc['name']}:"
    )
    return system, prompt


def resolve_world_report(
    doc: dict, node_outputs: Optional[dict] = None
) -> tuple[Optional[dict], list[str]]:
    node_outputs = node_outputs or {}
    warnings: list[str] = []
    picked = most_downstream(doc, node_outputs, WORLD_SOURCE_TYPES, "world")
    raw: Optional[dict] = None
    if picked is not None and isinstance(picked[1], dict):
        raw = picked[1]
    else:
        for n in doc.get("nodes") or []:
            if n["type"] != "world.spec" or n.get("disabled"):
                continue
            ins = input_values(doc, node_outputs, n["id"], "world")
            base = (
                ins[0] if ins and isinstance(ins[0], dict) else (n.get("params") or {}).get("world")
            )
            if isinstance(base, dict):
                raw = dict(base)
                for ts in input_values(doc, node_outputs, n["id"], "tileset"):
                    if isinstance(ts, dict):
                        raw["tileset"] = {**(raw.get("tileset") or {}), **ts}
                break
    if raw is None:
        return None, ["This flow has no world yet: add a World node, or run a World Generator."]
    raw = dict(raw)
    # NPC nodes wired anywhere in the flow join the world.
    extra = []
    for n in doc.get("nodes") or []:
        if n["type"] == "world.npc" and not n.get("disabled"):
            v = node_value(doc, node_outputs, n["id"], "npc")
            if not isinstance(v, dict):
                p = n.get("params") or {}
                sprites = input_values(doc, node_outputs, n["id"], "sprite")
                v = {
                    "name": p.get("name"),
                    "persona": p.get("persona"),
                    "greeting": p.get("greeting"),
                    "sprite": sprites[0] if sprites else None,
                }
            if not any(
                str(x.get("name", "")).lower() == str(v.get("name", "")).lower()
                for x in raw.get("npcs") or []
            ):
                extra.append(v)
    if extra:
        raw["npcs"] = list(raw.get("npcs") or []) + extra
    world = normalize_world(raw, seed = int(raw.get("seed") or doc.get("seed") or 0))
    eps = dict(world.get("entry_points") or {})
    for n in doc.get("nodes") or []:
        name = WORLD_ENTRY_TYPES.get(n["type"])
        if name and name not in eps and not n.get("disabled"):
            eps[name] = n["id"]
    world["entry_points"] = eps
    if "npc_reply" not in eps:
        warnings.append("No NPC Reply (live) node: NPCs will only say their greeting")
    return world, warnings


def resolve_world(doc: dict, node_outputs: Optional[dict] = None) -> Optional[dict]:
    return resolve_world_report(doc, node_outputs)[0]
