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
WORLD_ENTRY_TYPES = {
    "world.npc_reply": "npc_reply",
    "world.npc_plan": "npc_plan",
    "world.npc_chat": "npc_chat",
}
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
            **_npc_life(n, ref, npc_ids),
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
    sim = _simulation(raw.get("simulation"), width, height, regions)
    if sim is not None:
        world["simulation"] = sim
    return WorldSpec.model_validate(world).model_dump(exclude_none = True)


def _memory_items(raw: Any) -> list[dict]:
    out = []
    for m in raw if isinstance(raw, list) else []:
        text = m.get("text") if isinstance(m, dict) else m
        if isinstance(text, str) and text.strip():
            imp = m.get("importance") if isinstance(m, dict) else None
            out.append({"text": text.strip()[:300], "importance": _clamp_int(imp, 1, 10, 5)})
    return out[:20]


def _npc_life(n: dict, ref: Any, npc_ids: set[str]) -> dict:
    """The optional simulation fields, present only when the source has them so other worlds stay identical."""
    out: dict[str, Any] = {}
    if isinstance(n.get("routine"), str) and n["routine"].strip():
        out["routine"] = n["routine"].strip()[:600]
    if isinstance(n.get("relationships"), dict):
        rel = {}
        for k, v in n["relationships"].items():
            other = ref(k, npc_ids)
            if other and other != n["id"] and isinstance(v, str) and v.strip():
                rel[other] = v.strip()[:200]
        out["relationships"] = rel
    if "seed_memories" in n:
        out["seed_memories"] = _memory_items(n["seed_memories"])
    return out


def _simulation(raw: Any, width: int, height: int, regions: list[dict]) -> Optional[dict]:
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        return None
    taken: set[str] = set()
    places = []
    for p in raw.get("places") or []:
        if not isinstance(p, dict) or not str(p.get("name") or "").strip():
            continue
        x, y, w, h = (_clamp_int(p.get(k), 0, 256, -1) for k in ("x", "y", "w", "h"))
        if min(x, y) < 0 or min(w, h) < 1:
            continue
        x, y = min(x, width - 2), min(y, height - 2)
        places.append(
            {
                "id": _unique(slug(p.get("id") or p["name"], "place"), taken),
                "name": str(p["name"])[:60],
                "x": x,
                "y": y,
                "w": max(1, min(width - x, w)),
                "h": max(1, min(height - y, h)),
            }
        )
    if not places:
        places = [{k: r[k] for k in ("id", "name", "x", "y", "w", "h")} for r in regions]
    if not places:
        cx, cy = width // 2, height // 2
        places = [
            {"id": "town_square", "name": "Town square", "x": cx - 3, "y": cy - 2, "w": 7, "h": 5}
        ]
    num = raw.get("minutes_per_second")
    return {
        "enabled": raw.get("enabled", True) is not False,
        "minutes_per_second": max(0.1, min(60.0, float(num)))
        if isinstance(num, (int, float)) and not isinstance(num, bool)
        else 2.0,
        "start_minute": _clamp_int(raw.get("start_minute"), 0, 24 * 60 - 1, 480),
        "places": places[:16],
        "chat_cooldown_min": _clamp_int(raw.get("chat_cooldown_min"), 0, 7 * 24 * 60, 120),
        "calls_per_minute": _clamp_int(raw.get("calls_per_minute"), 0, 30, 4),
    }


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
NOTES_SCHEMA = {
    "type": "array",
    "maxItems": 2,
    "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "importance": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["text", "importance"],
    },
}
NPC_MEMORY_REPLY_SCHEMA = {
    **NPC_REPLY_SCHEMA,
    "properties": {**NPC_REPLY_SCHEMA["properties"], "remember": NOTES_SCHEMA},
}


def has_memory(state: dict) -> bool:
    """The world player sends `memory` only for simulated worlds; without it every prompt is the classic one."""
    return isinstance(state.get("memory"), list)


def clock_text(minutes: Any) -> str:
    m = int(minutes) if isinstance(minutes, (int, float)) and not isinstance(minutes, bool) else 0
    m = max(0, m)
    return f"Day {m // 1440 + 1}, {m % 1440 // 60:02d}:{m % 60:02d}"


def clean_notes(raw: Any, limit: int = 2) -> list[dict]:
    out = []
    for n in raw if isinstance(raw, list) else []:
        text = n.get("text") if isinstance(n, dict) else n
        if isinstance(text, str) and text.strip():
            imp = n.get("importance") if isinstance(n, dict) else None
            out.append(
                {
                    "text": re.sub(r"\s+", " ", text).strip()[:300],
                    "importance": _clamp_int(imp, 1, 10, 4),
                }
            )
    return out[:limit]


def npc_life(world: dict, npc: dict) -> str:
    names = {n["id"]: n["name"] for n in world.get("npcs") or []}
    lines = []
    if npc.get("routine"):
        lines.append(f"Usual routine: {npc['routine']}")
    for other, text in (npc.get("relationships") or {}).items():
        if other in names:
            lines.append(f"About {names[other]}: {text}")
    return "".join(f"{line}\n" for line in lines)


def memory_lines(records: list[dict]) -> str:
    return "\n".join(f"- {r['text']}" for r in records if r.get("text")) or "- nothing yet"


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
            if need:
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
    if has_memory(state):
        out["remember"] = clean_notes(raw.get("remember"))
    return out


def npc_prompt(
    state: dict,
    rules: str,
    memories: Optional[list[dict]] = None,
) -> tuple[str, str]:
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
    if memories is None:
        return system, prompt
    system += (
        f"\n{npc_life(world, npc)}"
        "`remember` lists 1-2 short notes, written as you, worth keeping from this exchange, each with an "
        "importance from 1 (mundane) to 10 (life changing)."
    )
    prompt = (
        f"It is {clock_text(state.get('clock'))}.\nWhat you remember:\n{memory_lines(memories)}\n"
        + prompt
    )
    return system, prompt


def plan_prompt(world: dict, npc: dict, memories: list[dict], day: int) -> tuple[str, str]:
    places = [p["name"] for p in (world.get("simulation") or {}).get("places") or []]
    system = f"You plan the day of {npc['name']}, a character in the game world \"{world.get('title')}\". Answer in JSON."
    prompt = (
        f"Persona: {npc.get('persona') or 'a friendly local'}\n{npc_life(world, npc)}"
        f"What {npc['name']} remembers:\n{memory_lines(memories)}\n"
        f"It is day {day + 1}. Plan {npc['name']}'s day from 07:00 to 22:00 in 4 to 6 broad blocks, each with a "
        f"start time (HH:MM), length in minutes, a short activity and a place, one of: {', '.join(places)}. "
        "Keep any appointments remembered above."
    )
    return system, prompt


def chat_schema(a: dict, b: dict) -> dict:
    return {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "minItems": 2,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string", "enum": [a["name"], b["name"]]},
                        "text": {"type": "string"},
                    },
                    "required": ["speaker", "text"],
                },
            },
            "a_learns": NOTES_SCHEMA,
            "b_learns": NOTES_SCHEMA,
        },
        "required": ["lines", "a_learns", "b_learns"],
    }


def chat_prompt(
    world: dict, a: dict, b: dict, mem_a: list[dict], mem_b: list[dict], state: dict
) -> tuple[str, str]:
    place = str(state.get("place") or "").strip()
    system = (
        f"You write a short chat between two characters in the game world \"{world.get('title')}\". "
        "Each stays in character and only shares what they know. Answer in JSON."
    )
    prompt = (
        f"It is {clock_text(state.get('clock'))}" + (f", at {place}" if place else "") + ".\n"
        f"{a['name']}: {a.get('persona') or 'a friendly local'}\n{npc_life(world, a)}"
        f"{a['name']} remembers:\n{memory_lines(mem_a)}\n\n"
        f"{b['name']}: {b.get('persona') or 'a friendly local'}\n{npc_life(world, b)}"
        f"{b['name']} remembers:\n{memory_lines(mem_b)}\n\n"
        f"{a['name']} and {b['name']} bump into each other. Write 2 to 6 short lines of their chat "
        f"(under 25 words each). `a_learns` is what {a['name']} newly learns from {b['name']} and `b_learns` what "
        f"{b['name']} learns from {a['name']}, written from the listener's point of view with an importance from 1 "
        "to 10; pass on news, plans and invitations they would share. Empty lists when nothing new comes up."
    )
    return system, prompt


def validate_chat(raw: Any, a: dict, b: dict) -> dict:
    if not isinstance(raw, dict):
        raise WorldError("the model did not return a chat")
    by_name = {a["name"].lower(): a["id"], b["name"].lower(): b["id"]}
    lines = []
    for line in raw.get("lines") or []:
        if not isinstance(line, dict):
            continue
        who = by_name.get(str(line.get("speaker") or "").strip().lower())
        text = re.sub(r"\s+", " ", str(line.get("text") or "")).strip()[:240]
        if who and text:
            lines.append({"npc": who, "text": text})
    if not lines:
        raise WorldError("the chat had no lines")
    return {
        "a": a["id"],
        "b": b["id"],
        "lines": lines[:6],
        "a_learns": clean_notes(raw.get("a_learns")),
        "b_learns": clean_notes(raw.get("b_learns")),
    }


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
    sim = world.get("simulation")
    if sim and sim.get("enabled") and not ({"npc_plan", "npc_chat"} & set(eps)):
        warnings.append(
            "Simulation is on but no NPC Plan or NPC Chat node: villagers keep wandering"
        )
    return world, warnings


def resolve_world(doc: dict, node_outputs: Optional[dict] = None) -> Optional[dict]:
    return resolve_world_report(doc, node_outputs)[0]
