# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""World specs: pick the world a flow produced, repair LLM-written worlds and validate NPC replies.

The map itself (terrain, paths, where things stand) is generated in the player from `seed` and `biome`
(`studio/frontend/src/features/flows/player/world/procgen.ts`), so a world spec stays small and shareable."""

from __future__ import annotations

import re
from typing import Any, Optional

from core.flows import assets as asset_store
from core.flows.city import normalize_city
from core.flows.authoring import strict_schema, string_enum
from core.flows.expr import Prng, seed_from
from core.flows.rivals import normalize_rivals
from core.flows.schema import WorldSpec
from core.flows.story import _unique, input_values, most_downstream, node_value, slug

WORLD_SOURCE_TYPES = ("world.play", "world.spec", "world.art", "world.generator")
WORLD_ENTRY_TYPES = {
    "world.npc_reply": "npc_reply",
    "world.npc_plan": "npc_plan",
    "world.npc_chat": "npc_chat",
    "world.describe_place": "describe_place",
}
BIOMES = ("meadow", "forest", "desert", "snow", "town", "dungeon", "beach")
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
# Keep in sync with MAX_HISTORY in world-state.ts.
MAX_HISTORY = 24
AUTONOMY_FLAGS = ("trade", "homes", "quests", "rivals", "enemies")
AUTONOMY_LEVELS = {
    "off": (),
    "village": ("trade", "homes"),
    "rivals": ("trade", "homes", "quests", "rivals"),
    "full": AUTONOMY_FLAGS,
}
NPC_ROLES = ("rival", "hostile")
ACTION_FLAGS = ("movement", "shooter")
GAMEPLAY_LEVELS = {
    "explore": {},
    "action": {"movement": True},
    "shooter": {"movement": True, "shooter": True, "verbs": True},
}


def action_of(raw: Any) -> Optional[dict]:
    """The world's action flags, or None when none is on (a shooter always moves with physics).
    Opt-in extras (``verbs``) appear only when set, so older worlds keep their exact shape."""
    if not isinstance(raw, dict):
        return None
    flags = {k: raw.get(k) is True for k in ACTION_FLAGS}
    flags["movement"] = flags["movement"] or flags["shooter"] or raw.get("verbs") is True
    if raw.get("verbs") is True:
        flags["verbs"] = True
    if not any(flags.values()):
        return None
    if raw.get("ai") == "v2":
        flags["ai"] = "v2"
        barks = _barks(raw.get("barks"))
        if barks:
            flags["barks"] = barks
    return flags


BARK_KINDS = ("see", "hear", "lose", "pain", "flee", "attack")
RAID_BARK_EVENTS = {
    "spot": "see",
    "lost": "lose",
    "hurt": "pain",
    "flank": "attack",
    "retreat": "flee",
}


def _barks(raw: Any) -> dict:
    """A world's own bark lines for hostile AI v2: known events only, up to 12 short lines each."""
    if isinstance(raw, list):  # a raid's tagged lines: [{"event": "spot", "text": ...}]
        by: dict = {}
        for b in raw:
            kind = RAID_BARK_EVENTS.get(b.get("event")) if isinstance(b, dict) else None
            if kind:
                by.setdefault(kind, []).append(b.get("text"))
        raw = by
    if not isinstance(raw, dict):
        return {}
    out = {}
    for kind in BARK_KINDS:
        lines = raw.get(kind)
        if not isinstance(lines, list):
            continue
        kept = [str(s).strip()[:80] for s in lines if isinstance(s, str) and s.strip()][:12]
        if kept:
            out[kind] = kept
    return out


MAX_ADDED_ROLE = 2
SUMMARY_CHARS = 600


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
            **({"role": n["role"]} if n.get("role") in NPC_ROLES else {}),
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
    auto = raw.get("autonomy")
    if isinstance(auto, dict) and any(auto.get(k) is True for k in AUTONOMY_FLAGS):
        world["autonomy"] = {k: auto.get(k) is True for k in AUTONOMY_FLAGS}
    act = action_of(raw.get("action"))
    if act:
        world["action"] = act
    if raw.get("render") in ("2d", "3d"):
        world["render"] = raw["render"]
    view = _view3d(raw.get("view3d"))
    if view:
        world["view3d"] = view
    raw_props = raw.get("props3d")
    if isinstance(raw_props, list):
        # Item links name items as the author wrote them; map them to the normalised ids.
        raw_props = [_prop_item_refs(p, lambda v: ref(v, item_ids)) for p in raw_props]
    props = props3d(raw_props, width, height, {i["id"] for i in world["items"]})
    if props:
        world["props3d"] = props
    rivals = normalize_rivals(raw.get("rivals"), npc_ids, regions, ref, taken)
    if rivals:
        world["rivals"] = rivals
    city = normalize_city(raw.get("city"))
    if city:
        # One side panel at a time: Mayor mode wins over Village Rivals (normalised above).
        world["city"] = city
        world.pop("rivals", None)
    return WorldSpec.model_validate(world).model_dump(exclude_none = True)


TIMES_OF_DAY = ("dawn", "day", "dusk", "night")
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _view3d(raw: Any) -> dict:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    if raw.get("time_of_day") in TIMES_OF_DAY:
        out["time_of_day"] = raw["time_of_day"]
    if isinstance(raw.get("sky"), str) and _HEX.match(raw["sky"].strip()):
        out["sky"] = raw["sky"].strip().lower()
    for key, lo, hi in (
        ("fog", 0.0, 1.0),
        ("camera_height", 0.3, 4.0),
        ("walk_speed", 0.5, 20.0),
        ("terrain", 0.0, 3.0),
    ):
        v = raw.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
            out[key] = round(max(lo, min(hi, float(v))), 3)
    if isinstance(raw.get("ambient"), dict):
        out["ambient"] = raw["ambient"]
    return out


PROP_KINDS = (
    "house",
    "tree",
    "lamp",
    "fence",
    "well",
    "crate",
    "rock",
    "bench",
    "stall",
    "door",
    "chest",
    "lever",
    "model",
)
OPEN_PROPS = ("lamp", "bench", "lever")
MAX_PROP_LINKS = 8
_ITEM_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
MAX_PROPS = 400
_PROP_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def props3d(
    raw: Any,
    width: int,
    height: int,
    item_ids: Optional[set] = None,
) -> list[dict]:
    """Clean the 3D editor's props: unknown kinds, duplicate ids and models without a flow asset are dropped,
    positions are kept on the map and rotation / scale clamped. Doors, chests and levers keep their links
    (key and contents item ids, when ``item_ids`` is given only known ones; lever targets only real doors)."""
    out: list[dict] = []
    seen: set[str] = set()
    for p in raw if isinstance(raw, list) else []:
        if len(out) >= MAX_PROPS:
            break
        if not isinstance(p, dict) or p.get("kind") not in PROP_KINDS:
            continue
        pid = p.get("id")
        if not isinstance(pid, str) or not _PROP_ID.match(pid) or pid in seen:
            continue
        x, y = p.get("x"), p.get("y")
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and v == v for v in (x, y)
        ):
            continue
        prop: dict[str, Any] = {"id": pid, "kind": p["kind"]}
        if p["kind"] == "model":
            asset = p.get("asset")
            if not (
                isinstance(asset, dict)
                and isinstance(asset.get("ref"), str)
                and asset_store.parse_ref(asset["ref"])
            ):
                continue
            prop["asset"] = {
                "kind": "model",
                "ref": asset["ref"],
                "mime": "model/gltf-binary",
                **({"url": asset["url"]} if isinstance(asset.get("url"), str) else {}),
            }
        if isinstance(p.get("name"), str) and p["name"].strip():
            prop["name"] = p["name"].strip()[:60]
        prop["x"] = round(max(0.0, min(width - 0.001, float(x))), 3)
        prop["y"] = round(max(0.0, min(height - 0.001, float(y))), 3)
        rot = p.get("rot")
        prop["rot"] = (
            round(float(rot) % 360, 2)
            if isinstance(rot, (int, float)) and not isinstance(rot, bool) and abs(rot) < 1e9
            else 0.0
        )
        scale = p.get("scale")
        prop["scale"] = (
            round(max(0.2, min(5.0, float(scale))), 3)
            if isinstance(scale, (int, float)) and not isinstance(scale, bool) and scale == scale
            else 1.0
        )
        prop["solid"] = (
            p["solid"] if isinstance(p.get("solid"), bool) else p["kind"] not in OPEN_PROPS
        )
        _prop_links(prop, p, item_ids)
        seen.add(pid)
        out.append(prop)
    doors = {p["id"] for p in out if p["kind"] == "door"}
    for prop in out:
        if "targets" in prop:
            prop["targets"] = [t for t in prop["targets"] if t in doors]
            if not prop["targets"]:
                del prop["targets"]
    return out


def _prop_item_refs(p: Any, resolve) -> Any:
    if not isinstance(p, dict):
        return p
    out = dict(p)
    if isinstance(p.get("locked_by"), str):
        out["locked_by"] = resolve(p["locked_by"]) or p["locked_by"]
    if isinstance(p.get("contents"), list):
        out["contents"] = [(resolve(v) or v) if isinstance(v, str) else v for v in p["contents"]]
    return out


def _prop_links(prop: dict, raw: dict, item_ids: Optional[set]) -> None:
    kind = prop["kind"]
    if kind not in ("door", "chest", "lever"):
        return

    def item(v: Any) -> bool:
        return (
            isinstance(v, str) and bool(_ITEM_ID.match(v)) and (item_ids is None or v in item_ids)
        )

    if isinstance(raw.get("open"), bool):
        prop["open"] = raw["open"]
    if kind == "door" and item(raw.get("locked_by")):
        prop["locked_by"] = raw["locked_by"]
    if kind == "chest" and isinstance(raw.get("contents"), list):
        contents = list(dict.fromkeys(v for v in raw["contents"] if item(v)))[:MAX_PROP_LINKS]
        if contents:
            prop["contents"] = contents
    if kind == "lever" and isinstance(raw.get("targets"), list):
        targets = list(
            dict.fromkeys(v for v in raw["targets"] if isinstance(v, str) and _PROP_ID.match(v))
        )[:MAX_PROP_LINKS]
        if targets:
            prop["targets"] = targets


def apply_play_view(
    world: dict,
    doc: dict,
    node_outputs: Optional[dict] = None,
    source: Optional[str] = None,
) -> dict:
    """Player node view settings win over the world's (read from the saved flow, so no rerun); the one fed by ``source`` wins."""
    players = [
        n for n in doc.get("nodes") or [] if n["type"] == "world.play" and not n.get("disabled")
    ]
    fed = {e["target"] for e in doc.get("edges") or [] if source and e.get("source") == source}
    players.sort(key = lambda n: (n["id"] != source, n["id"] not in fed))
    for n in players:
        p = n.get("params") or {}
        if p.get("view") in ("2d", "3d"):
            world["render"] = p["view"]
        view = dict(world.get("view3d") or {})
        if p.get("time_of_day") in TIMES_OF_DAY:
            view["time_of_day"] = p["time_of_day"]
        for a in input_values(doc, node_outputs or {}, n["id"], "ambient"):
            if isinstance(a, dict):
                view["ambient"] = a
        view = _view3d(view)
        if view:
            world["view3d"] = view
        if p.get("gameplay") in GAMEPLAY_LEVELS:
            # The world keeps its own hostile AI and barks under the player's gameplay level.
            own = {k: v for k, v in (world.get("action") or {}).items() if k in ("ai", "barks")}
            act = action_of({**GAMEPLAY_LEVELS[p["gameplay"]], **own})
            if act:
                world["action"] = act
            else:
                world.pop("action", None)
        if isinstance(p.get("props3d"), list):
            world["props3d"] = props3d(
                p["props3d"],
                world["width"],
                world["height"],
                {i["id"] for i in world.get("items") or []},
            )
        world["player_node"] = n["id"]
        break
    return world


def _memory_items(raw: Any) -> list[dict]:
    out = []
    for m in raw if isinstance(raw, list) else []:
        text = m.get("text") if isinstance(m, dict) else m
        if isinstance(text, str) and text.strip():
            imp = m.get("importance") if isinstance(m, dict) else None
            out.append({"text": text.strip()[:300], "importance": _clamp_int(imp, 1, 10, 5)})
    return out[:20]


def _npc_life(n: dict, ref: Any, npc_ids: set[str]) -> dict:
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
        "needs": raw.get("needs", True) is not False,
        **_life_fields(raw),
    }


def _life_fields(raw: dict) -> dict:
    out: dict[str, Any] = {}
    for flag in ("options", "relationships"):
        if raw.get(flag) is True:
            out[flag] = True
    for key, lo, hi in (("plan_cache", 1, 256), ("reflect_at", 10, 5000), ("rate_batch", 1, 32)):
        v = raw.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            out[key] = _clamp_int(v, lo, hi, lo)
    return out


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


REPLY_ACTIONS = ("follow_player", "stop_following", "go_to")
FOLLOW_ASK = re.compile(r"\b(follow me|come with me|come along|walk with me|tag along)\b")
FOLLOW_STOP = re.compile(
    r"\b(stop following|go (back )?home|you can go|wait (for me )?here|stay here"
    r"|(don['\u2019]?t|do not|never|no need to) (follow|come with|come along|walk with|tag along))\b"
)
TRADE_ASK = re.compile(r"\b(buy|sell|trade|for sale|wares|what do you have|how much)\b")

NPC_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "give_item": {"type": "string"},
        "take_item": {"type": "string"},
        "complete_quest": {"type": "string"},
        "action": {"type": "string", "enum": ["", *REPLY_ACTIONS]},
        "place": {"type": "string"},
        "offer_trade": {"type": "string"},
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


SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


DIRECTOR_SCHEMA = strict_schema(
    {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "reward_text": {"type": "string"},
            "giver": {"type": "string"},
            "item": {"type": "string"},
            "new_item_name": {"type": "string"},
            "new_item_description": {"type": "string"},
            "new_npc_name": {"type": "string"},
            "new_npc_persona": {"type": "string"},
            "new_npc_greeting": {"type": "string"},
            "new_npc_role": {"type": "string", "enum": ["", *NPC_ROLES]},
            "region": {"type": "string"},
        },
        "required": [
            "title",
            "description",
            "reward_text",
            "giver",
            "item",
            "new_item_name",
            "new_npc_name",
        ],
    }
)


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


def heard_from(record: dict, names: Optional[dict] = None) -> str:
    chain = [
        (names or {}).get(s, "the player" if s == "player" else s)
        for s in record.get("source") or []
        if isinstance(s, str) and s
    ][:4]
    if not chain:
        return ""
    return " (heard from " + ", who heard it from ".join(chain) + ")"


def memory_lines(records: list[dict], names: Optional[dict] = None) -> str:
    return (
        "\n".join(f"- {r['text']}{heard_from(r, names)}" for r in records if r.get("text"))
        or "- nothing yet"
    )


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
    wanted = {
        q.get("requires_item")
        for q in world.get("quests") or []
        if q["giver"] == npc["id"] and q["id"] not in done
    }
    if give in item_ids and give not in inventory and give not in wanted:
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
    if "complete_quest" not in out:
        for q in world.get("quests") or []:
            need = q.get("requires_item")
            if q["giver"] == npc["id"] and q["id"] not in done and need and need in inventory:
                out["complete_quest"] = q["id"]
                out["take_item"] = need
                break
    sets = {}
    declared = world.get("vars") or {}
    for s in raw.get("set") or []:
        if isinstance(s, dict) and s.get("var") in declared:
            v = s.get("value")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                sets[s["var"]] = v
    if sets:
        out["set"] = sets
    action = str(raw.get("action") or "").strip()
    place = str(raw.get("place") or "").strip()
    asked = str(state.get("player_message") or "").lower()
    named = {
        p["id"]
        for p in world_places(world)
        if p["id"].replace("_", " ") in asked
        or any(
            len(w) > 3 and w in asked
            for w in re.findall(r"[a-z]+", str(p.get("name") or "").lower())
        )
    }
    if npc.get("wander") is False:
        pass  # A villager who stays put (a shopkeeper at their stall) takes no movement action.
    elif FOLLOW_STOP.search(asked):
        out["action"] = "stop_following"
    elif FOLLOW_ASK.search(asked):
        out["action"] = "follow_player"
    elif action == "go_to" and place in named:
        out["action"], out["place"] = action, place
    elif action in ("follow_player", "stop_following"):
        out["action"] = action
    goods = [g["id"] for g in state.get("goods") or [] if isinstance(g, dict) and g.get("id")]
    offer = str(raw.get("offer_trade") or "").strip().lower()
    if offer in goods:
        out["offer_trade"] = offer
    elif goods and TRADE_ASK.search(asked):
        out["offer_trade"] = goods[0]
    if has_memory(state):
        out["remember"] = clean_notes(raw.get("remember"))
    return out


def talk_summary(state: dict) -> tuple[str, int]:
    s = state.get("summary") if isinstance(state.get("summary"), dict) else {}
    text = re.sub(r"\s+", " ", str(s.get("text") or "")).strip()[:SUMMARY_CHARS]
    upto = s.get("upto")
    return text, upto if isinstance(upto, int) and not isinstance(upto, bool) else -1


def compact_prompt(state: dict) -> tuple[str, str]:
    world, npc = npc_context(state)
    player = (world.get("player") or {}).get("name", "Player")
    previous, _ = talk_summary(state)
    turns = "\n".join(
        f"{player if t.get('role') == 'player' else npc['name']}: {str(t.get('text') or '')[:400]}"
        for t in state.get("turns") or []
        if isinstance(t, dict)
    )
    system = (
        f"You keep the memory of {npc['name']}, a character in the game world \"{world.get('title')}\".\n"
        f"Persona: {npc.get('persona') or 'a friendly local'}\n"
        f"Write `summary`: what {npc['name']} remembers of talking with {player}, in first person, under "
        f"{SUMMARY_CHARS} characters. Keep names, promises, facts {player} shared, items and quests, and how "
        f"{npc['name']} feels about {player}. Drop small talk. Answer in JSON."
    )
    prompt = (
        f"What you remembered so far: {previous or 'nothing yet'}\n"
        f"Conversation to fold in:\n{turns or '(nothing)'}\n"
    )
    return system, prompt


def clean_summary(raw: Any) -> str:
    text = raw.get("summary") if isinstance(raw, dict) else raw
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        raise WorldError("the model did not return a summary")
    if len(text) > SUMMARY_CHARS:
        cut = text[:SUMMARY_CHARS]
        end = cut.rfind(". ")
        text = cut[: end + 1] if end > SUMMARY_CHARS // 2 else cut
    return text


def _used_items(world: dict, state: dict) -> set:
    return (
        {q.get("requires_item") for q in world.get("quests") or []}
        | set(state.get("inventory") or [])
        | set(state.get("picked") or [])
    )


def director_prompt(state: dict) -> tuple[str, str]:
    world = state.get("world") if isinstance(state.get("world"), dict) else None
    if world is None:
        raise WorldError("the director needs the world in its state")
    done = set(state.get("completed_quests") or [])
    npcs = "\n".join(f"- {n['id']}: {n['name']}" for n in world.get("npcs") or [])
    used = _used_items(world, state)
    items = "\n".join(
        f"- {i['id']}: {i['name']}" for i in world.get("items") or [] if i["id"] not in used
    )
    regions = "\n".join(f"- {r['id']}: {r['name']}" for r in world.get("regions") or [])
    quests = "\n".join(
        f"- {q['title']} ({'done' if q['id'] in done else 'open'})"
        for q in world.get("quests") or []
    )
    events = "; ".join(str(e)[:160] for e in (state.get("events") or [])[-6:] if isinstance(e, str))
    rules = [
        "Make it fit the story so far and not repeat a quest.",
        "`giver` is a villager id from the list.",
        "`item` is the id of an item from the list that the quest asks the player to bring.",
    ]
    if state.get("allow_item"):
        rules.append(
            "Or invent one new item for it (new_item_name, new_item_description) and leave `item` empty."
            if items
            else "Every item is already used, so invent one new item for it (new_item_name, "
            "new_item_description) and leave `item` empty."
        )
    if state.get("allow_npc"):
        rules.append(
            "Or invent one new villager to give it (new_npc_name, new_npc_persona, new_npc_greeting) and set "
            "`giver` to their name."
        )
    roles = {
        "rival": "rival (races the player for quest items)",
        "hostile": "hostile (chases and robs the player, never hurts anyone)",
    }
    allowed = [roles[r] for r in NPC_ROLES if state.get("allow_npc") and state.get(f"allow_{r}")]
    if allowed:
        rules.append(
            "A new villager may set `new_npc_role` to "
            + " or ".join(allowed)
            + " when it suits the story; leave it empty for a friendly villager."
        )
    if state.get("allow_item") or state.get("allow_npc"):
        rules.append("`region` is the region id where anything new appears.")
    system = (
        f"You write the next quest for the game world \"{world.get('title')}\". {world.get('intro_text') or ''}\n"
        + " ".join(rules)
        + " Leave fields you do not use empty. Answer in JSON."
    )
    prompt = (
        f"Villagers:\n{npcs or '- none'}\nItems:\n{items or '- none'}\nRegions:\n{regions or '- none'}\n"
        f"Quests so far:\n{quests or '- none'}\nRecent events: {events or 'none'}\n"
    )
    return system, prompt


def validate_director(raw: Any, state: dict) -> dict:
    world = state.get("world") if isinstance(state.get("world"), dict) else None
    if world is None or not isinstance(raw, dict):
        raise WorldError("the director did not return a quest")

    def text(key: str, limit: int) -> str:
        t = re.sub(r"\s+", " ", str(raw.get(key) or "")).strip()
        if len(t) <= limit:
            return t
        cut = t[:limit]
        end = cut.rfind(". ")
        return cut[: end + 1] if end > limit // 2 else cut

    title = text("title", 120)
    if not title:
        raise WorldError("the new quest has no title")
    npcs = world.get("npcs") or []
    items = world.get("items") or []
    quests = world.get("quests") or []
    regions = world.get("regions") or []
    taken = {
        x["id"] for x in [*npcs, *items, *quests, *regions] if isinstance(x, dict) and x.get("id")
    }
    region_ids = {r["id"] for r in regions}
    region = text("region", 80)
    region = region if region in region_ids else (regions[0]["id"] if regions else None)

    def find(pool: list[dict], key: str) -> Optional[str]:
        k = key.strip().lower()
        hit = next(
            (x["id"] for x in pool if x["id"].lower() == k or str(x["name"]).lower() == k), None
        )
        return hit or next((x["id"] for x in pool if x["id"] == slug(k)), None)

    patch: dict[str, Any] = {"npcs": [], "items": []}
    giver = find(npcs, text("giver", 80))
    new_npc_name = text("new_npc_name", 60)
    if giver is None and new_npc_name and state.get("allow_npc"):
        giver = _unique(slug(new_npc_name, "npc"), taken)
        patch["npcs"].append(
            {
                "id": giver,
                "name": new_npc_name,
                "persona": text("new_npc_persona", 600) or "a friendly newcomer",
                "greeting": text("new_npc_greeting", 300) or f"Hello, I'm {new_npc_name}.",
                **({"region": region} if region else {}),
                **(
                    {"role": role}
                    if (role := text("new_npc_role", 20)) in NPC_ROLES
                    and state.get(f"allow_{role}")
                    else {}
                ),
            }
        )
    if giver is None:
        raise WorldError("the new quest has no giver in the world")
    used = _used_items(world, state)
    asked = text("item", 80)
    need = find(items, asked)
    if need in used:
        need = None
    new_item_name = text("new_item_name", 60) or (
        asked[:60] if asked and find(items, asked) is None else ""
    )
    if re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)+", new_item_name):
        new_item_name = new_item_name.replace("_", " ").capitalize()
    if need is None and new_item_name and state.get("allow_item"):
        need = _unique(slug(new_item_name, "item"), taken)
        patch["items"].append(
            {
                "id": need,
                "name": new_item_name,
                "description": text("new_item_description", 300),
                **({"region": region} if region else {}),
            }
        )
    if need is None:
        raise WorldError("the new quest needs an item that is in the world and not used yet")
    patch["quest"] = {
        "id": _unique(slug(title, "quest"), taken),
        "title": title,
        "description": text("description", 500),
        "giver": giver,
        "requires_item": need,
        "reward_text": text("reward_text", 200) or "Thank you!",
    }
    return patch


def director_schema(state: dict) -> dict:
    """DIRECTOR_SCHEMA with the giver and item fields limited to what the world has, when nothing new may be invented."""
    world = state.get("world") if isinstance(state.get("world"), dict) else {}
    props = dict(DIRECTOR_SCHEMA["properties"])
    if not state.get("allow_npc"):
        props["giver"] = string_enum(n.get("id") for n in world.get("npcs") or [])
    if not state.get("allow_item"):
        used = _used_items(world, state)
        props["item"] = string_enum(
            i.get("id") for i in world.get("items") or [] if i.get("id") not in used
        )
    regions = [r.get("id") for r in world.get("regions") or []]
    if regions:
        props["region"] = string_enum(regions, allow_empty = True)
    return strict_schema({**DIRECTOR_SCHEMA, "properties": props})


def fallback_quest(state: dict) -> dict:
    """A plain fetch quest from what the world already has, when the model's answers keep being refused."""
    world = state.get("world") if isinstance(state.get("world"), dict) else {}
    npcs = world.get("npcs") or []
    if not npcs:
        raise WorldError("the world has no villager to give a quest")
    used = _used_items(world, state)
    free = [i for i in world.get("items") or [] if i.get("id") not in used]
    n = len(world.get("quests") or [])
    giver = npcs[n % len(npcs)]
    if free:
        item = free[n % len(free)]
        raw = {"item": item["id"], "new_item_name": ""}
        name = item["name"]
    elif state.get("allow_item"):
        name = FALLBACK_ITEMS[n % len(FALLBACK_ITEMS)]
        raw = {
            "item": "",
            "new_item_name": name,
            "new_item_description": f"A {name.lower()} someone lost.",
        }
    else:
        raise WorldError("every item is already used and new ones are off")
    return validate_director(
        {
            **raw,
            "title": f"Find the {name}",
            "description": f"{giver['name']} has lost the {name} and would be glad to have it back.",
            "reward_text": "Thank you, that means a lot!",
            "giver": giver["id"],
            "new_npc_name": "",
        },
        state,
    )


FALLBACK_ITEMS = (
    "Brass Key",
    "Old Map",
    "Silver Locket",
    "Lucky Coin",
    "Lost Letter",
    "Little Bell",
)


def world_places(world: dict) -> list[dict]:
    seen, out = set(), []
    for p in [
        *((world.get("simulation") or {}).get("places") or []),
        *(world.get("regions") or []),
    ]:
        if isinstance(p, dict) and p.get("id") and p["id"] not in seen:
            seen.add(p["id"])
            out.append(p)
    return out


def action_rules(world: dict) -> str:
    places = ", ".join(f"{p['name']} (id {p['id']})" for p in world_places(world)[:12])
    return (
        "\nSet `action` to follow_player when you agree to come along with the player, stop_following when you "
        "part ways"
        + (
            f", or go_to with a `place` id when you agree to head somewhere: {places}"
            if places
            else ""
        )
        + ". Leave it empty otherwise."
    )


ROLE_LINES = {
    "rival": (
        "Your role: the player's RIVAL. You want the same quest items they want and mean to find them first: brag "
        "about it, tease them, and never help them with a quest or tell them where an item is. Never cruel."
    ),
    "hostile": (
        "Your role: HOSTILE to the player. Speak gruffly and threateningly, refuse to help them, and demand coins "
        "to leave them alone. You never hurt anyone."
    ),
}


def role_line(state: dict) -> str:
    role = state.get("role")
    return f"{ROLE_LINES[role]}\n" if role in ROLE_LINES else ""


def trade_rules(state: dict) -> str:
    goods = [g for g in state.get("goods") or [] if isinstance(g, dict) and g.get("id")]
    if not goods:
        return ""
    wares = ", ".join(
        f"{g.get('name') or g['id']} (id {g['id']}, {g.get('price')} coins)" for g in goods[:7]
    )
    return (
        f"\nYou sell: {wares}. When the player wants to buy, set `offer_trade` to one of those ids; the "
        "player decides in the trade panel."
    )


def npc_prompt(
    state: dict,
    rules: str,
    memories: Optional[list[dict]] = None,
) -> tuple[str, str]:
    world, npc = npc_context(state)
    summary, upto = talk_summary(state)
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
        + role_line(state)
        + f"Persona: {npc.get('persona') or 'a friendly local'}\n"
        f"World: {world.get('intro_text') or ''}\n"
        + (
            f"Earlier conversations with the player, as you remember them: {summary}\n"
            if summary
            else ""
        )
        + f"{rules}\n"
        "Reply as JSON. `text` is what you say. Only set `complete_quest` to one of your quest ids when the player has "
        "earned it (they carry the needed item). Use `give_item` only for an item id from the world list that fits the "
        "story; use empty strings when nothing happens."
    )
    if npc.get("wander") is not False:
        system += action_rules(world)
    system += trade_rules(state)
    system += f"\n{role_line(state)}" if role_line(state) else ""
    # Summary lives in the system prompt (changes only on fold) so llama-server reuses the cached prefix.
    history = [
        h
        for h in state.get("history") or []
        if isinstance(h, dict) and not (isinstance(h.get("seq"), int) and h["seq"] <= upto)
    ][-MAX_HISTORY:]
    convo = "\n".join(
        f"{'Player' if h.get('role') == 'player' else npc['name']}: {str(h.get('text') or '')[:400]}"
        for h in history
    )
    item_list = ", ".join("{} (id {})".format(i["name"], i["id"]) for i in items.values())
    quest_list = "\n".join(q_lines)
    # The conversation comes first: what changes between turns (quests, inventory, variables) follows it, so a
    # picked-up item does not make llama-server process the whole conversation again.
    head = (
        f"Conversation so far:\n{convo or '(just met)'}\n"
        f"Your quests:\n{quest_list or '- none'}\n"
        f"World items: {item_list or 'none'}\n"
        f"Player ({(world.get('player') or {}).get('name', 'Player')}) carries: {', '.join(inv) or 'nothing'}\n"
        f"Variables: {state.get('vars') or {}}\n"
        + ("You are walking with the player.\n" if state.get("following") else "")
    )
    turn = f"Player: {str(state.get('player_message') or '')[:600]}\n{npc['name']}:"
    if memories is None:
        return system, head + turn
    system += (
        f"\n{npc_life(world, npc)}"
        "`remember` lists 1-2 short notes, written as you, worth keeping from this exchange, each with an "
        "importance from 1 (mundane) to 10 (life changing)."
    )
    # Per-turn parts go last so llama-server reuses the cached prefix.
    return system, (
        head
        + f"It is {clock_text(state.get('clock'))}.\nWhat you remember:\n"
        + f"{memory_lines(memories, {n['id']: n['name'] for n in world.get('npcs') or []})}\n"
        + turn
    )


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


NEED_NAMES = ("energy", "hunger", "social", "fun")
OPTION_ACTIONS = ("go_to", "eat", "sleep", "work", "socialise", "wander")
UNTIL_KINDS = ("time", "need", "item", "place")
MAX_OPTIONS = 6
DAY_END = 22 * 60


def options_schema(places: list[str], items: list[str]) -> dict:
    """Flat fields so small models fill them; unused ones stay empty."""
    return {
        "type": "object",
        "properties": {
            "options": {
                "type": "array",
                "maxItems": MAX_OPTIONS,
                "items": {
                    "type": "object",
                    "properties": {
                        "activity": {"type": "string"},
                        "place": {"type": "string", "enum": places},
                        "action": {"type": "string", "enum": list(OPTION_ACTIONS)},
                        "until": {"type": "string", "enum": list(UNTIL_KINDS)},
                        "at": {"type": "string", "pattern": "^([01][0-9]|2[0-3]):[0-5][0-9]$"},
                        "need": {"type": "string", "enum": list(NEED_NAMES)},
                        "above": {"type": "integer", "minimum": 0, "maximum": 100},
                        "item": {"type": "string", "enum": ["", *items]},
                    },
                    "required": [
                        "activity",
                        "place",
                        "action",
                        "until",
                        "at",
                        "need",
                        "above",
                        "item",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["options"],
        "additionalProperties": False,
    }


def options_prompt(
    world: dict, npc: dict, memories: list[dict], clock: Any, needs: Any
) -> tuple[str, str]:
    places = [p["name"] for p in (world.get("simulation") or {}).get("places") or []]
    names = {n["id"]: n["name"] for n in world.get("npcs") or []}
    feel = ""
    if isinstance(needs, dict):
        feel = (
            "Needs (0 empty, 100 full): "
            + ", ".join(
                f"{k} {int(needs[k])}" for k in NEED_NAMES if isinstance(needs.get(k), (int, float))
            )
            + "\n"
        )
    system = (
        f"You plan what {npc['name']} does next, a character in the game world \"{world.get('title')}\". "
        "Answer in JSON."
    )
    # Stable persona and rules first so the server reuses the cached prefix; memories, needs and time last.
    prompt = (
        f"Persona: {npc.get('persona') or 'a friendly local'}\n{npc_life(world, npc)}"
        f"List 3 to {MAX_OPTIONS} things {npc['name']} does next, in order, until 22:00. "
        f"Each has an activity, a place (one of: {', '.join(places)}), an action ({', '.join(OPTION_ACTIONS)}) and "
        "when it ends: `until` time (at `at`, HH:MM), need (when `need` rises above `above`), item (once they hold "
        "`item`) or place (once they reach the place); fill unused fields with \"\" or 0. Keep any appointments "
        "remembered below.\n\n"
        f"What {npc['name']} remembers:\n{memory_lines(memories, names)}\n{feel}"
        f"It is {clock_text(clock)}."
    )
    return system, prompt


def clean_options(raw: Any, world: dict, clock: Any) -> list[dict]:
    """Options with absolute times and ids; anything unknown is dropped, not guessed."""
    sim = world.get("simulation") or {}
    by_name = {p["name"]: p["id"] for p in sim.get("places") or []}
    items = {i["name"]: i["id"] for i in world.get("items") or []}
    now = int(clock) if isinstance(clock, (int, float)) and not isinstance(clock, bool) else 0
    day = max(0, now) // 1440 * 1440
    out = []
    for o in (raw.get("options") if isinstance(raw, dict) else None) or []:
        if len(out) >= MAX_OPTIONS:
            break
        if not isinstance(o, dict) or o.get("place") not in by_name:
            continue
        activity = re.sub(r"\s+", " ", str(o.get("activity") or "")).strip()[:80]
        action = o.get("action") if o.get("action") in OPTION_ACTIONS else "go_to"
        kind = o.get("until")
        until: Optional[dict] = None
        if kind == "time":
            try:
                hh, mm = (int(x) for x in str(o.get("at") or "").split(":"))
            except ValueError:
                continue
            at = day + hh * 60 + mm
            if 0 <= hh < 24 and 0 <= mm < 60 and at > now:
                until = {"type": "time", "at": at}
        elif kind == "need" and o.get("need") in NEED_NAMES:
            until = {
                "type": "need",
                "need": o["need"],
                "above": _clamp_int(o.get("above"), 0, 100, 80),
            }
        elif kind == "item" and o.get("item") in items:
            until = {"type": "item", "item": items[o["item"]]}
        elif kind == "place":
            until = {"type": "place", "place": by_name[o["place"]]}
        if until is None or not activity:
            continue
        out.append(
            {"activity": activity, "place": by_name[o["place"]], "kind": action, "until": until}
        )
    return out


def options_blocks(options: list[dict], clock: Any) -> list[dict]:
    """Plan blocks for the classic HUD and memory text: a timed option ends at its time, others take an hour."""
    t = int(clock) if isinstance(clock, (int, float)) and not isinstance(clock, bool) else 0
    blocks = []
    for o in options:
        end = o["until"]["at"] if o["until"]["type"] == "time" else t + 60
        if end <= t:
            continue
        blocks.append(
            {"start": t, "minutes": end - t, "activity": o["activity"], "place": o["place"]}
        )
        t = end
    return blocks


REFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "insights": {
            "type": "array",
            "maxItems": 3,
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


def reflect_prompt(world: dict, npc: dict, records: list[dict]) -> tuple[str, str]:
    names = {n["id"]: n["name"] for n in world.get("npcs") or []}
    listing = "\n".join(
        f"{i + 1}. {r['text']}{heard_from(r, names)}" for i, r in enumerate(records)
    )
    system = f"You help {npc['name']} reflect on their memories. Answer in JSON."
    prompt = (
        f"Persona: {npc.get('persona') or 'a friendly local'}\n"
        f"What 1 to 3 high-level insights can {npc['name']} infer from the statements below? For each, list the "
        "numbers of the statements it rests on and rate its importance from 1 to 10.\n\n"
        f"Statements:\n{listing}"
    )
    return system, prompt


def clean_insights(raw: Any, records: list[dict]) -> list[dict]:
    out = []
    for ins in (raw.get("insights") if isinstance(raw, dict) else None) or []:
        if not isinstance(ins, dict):
            continue
        text = re.sub(r"\s+", " ", str(ins.get("text") or "")).strip()[:300]
        if not text:
            continue
        cites = [
            records[n - 1]["id"]
            for n in (ins.get("cites") if isinstance(ins.get("cites"), list) else [])
            if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(records)
        ]
        out.append(
            {
                "text": text,
                "importance": _clamp_int(ins.get("importance"), 1, 10, 6),
                "cites": cites,
            }
        )
    return out[:3]


def rel_line(name: str, other: str, rel: Any) -> str:
    if not isinstance(rel, dict):
        return ""
    aff = _clamp_int(rel.get("affinity"), -10, 10, 0)
    facts = [str(f)[:80] for f in rel.get("facts") or [] if isinstance(f, str) and f.strip()][:3]
    mood = "warm" if aff >= 3 else "cool" if aff <= -3 else "neutral"
    return (
        f"{name} feels {mood} towards {other} (affinity {aff:+d})"
        + (f"; knows: {'; '.join(facts)}" if facts else "")
        + "\n"
    )


SOCIAL_NOTES_SCHEMA = {
    **NOTES_SCHEMA,
    "items": {
        "type": "object",
        "properties": {
            **NOTES_SCHEMA["items"]["properties"],
            "from": {"type": "integer", "minimum": 0, "maximum": 10},
        },
        "required": ["text", "importance", "from"],
        "additionalProperties": False,
    },
}
FEELS_SCHEMA = {
    "type": "object",
    "properties": {
        "affinity": {"type": "integer", "minimum": -3, "maximum": 3},
        "fact": {"type": "string"},
    },
    "required": ["affinity", "fact"],
    "additionalProperties": False,
}


def chat_schema(
    a: dict,
    b: dict,
    social: bool = False,
) -> dict:
    if social:
        base = chat_schema(a, b)
        lines = base["properties"]["lines"]
        props = {
            **base["properties"],
            "lines": {**lines, "items": {**lines["items"], "additionalProperties": False}},
            "a_learns": SOCIAL_NOTES_SCHEMA,
            "b_learns": SOCIAL_NOTES_SCHEMA,
            "a_feels": FEELS_SCHEMA,
            "b_feels": FEELS_SCHEMA,
        }
        return {
            **base,
            "properties": props,
            "required": [*base["required"], "a_feels", "b_feels"],
            "additionalProperties": False,
        }
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
    world: dict,
    a: dict,
    b: dict,
    mem_a: list[dict],
    mem_b: list[dict],
    state: dict,
    social: bool = False,
) -> tuple[str, str]:
    place = str(state.get("place") or "").strip()
    system = (
        f"You write a short chat between two characters in the game world \"{world.get('title')}\". "
        "Each stays in character and only shares what they know. Answer in JSON."
    )
    if social:
        return system, _social_chat_prompt(world, a, b, mem_a, mem_b, state, place)
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


def _numbered(records: list[dict], names: dict) -> str:
    return (
        "\n".join(f"{i + 1}. {r['text']}{heard_from(r, names)}" for i, r in enumerate(records))
        or "- nothing yet"
    )


def _social_chat_prompt(
    world: dict, a: dict, b: dict, mem_a: list[dict], mem_b: list[dict], state: dict, place: str
) -> str:
    names = {n["id"]: n["name"] for n in world.get("npcs") or []}
    # Personas and rules first (cacheable per pair); feelings, memories, time and place last.
    return (
        f"{a['name']}: {a.get('persona') or 'a friendly local'}\n{npc_life(world, a)}"
        f"{b['name']}: {b.get('persona') or 'a friendly local'}\n{npc_life(world, b)}\n"
        f"{a['name']} and {b['name']} bump into each other. Write 2 to 6 short lines of their chat "
        f"(under 25 words each). `a_learns` is what {a['name']} newly learns from {b['name']} and `b_learns` what "
        f"{b['name']} learns from {a['name']}, written from the listener's point of view with an importance from 1 "
        "to 10 and `from` = the number of the speaker's memory it repeats (0 if new); pass on news, plans and "
        "invitations they would share. Empty lists when nothing new comes up. `a_feels` and `b_feels`: how the chat "
        "changed each one's liking of the other (-3 to 3) and one short fact they now know about the other "
        "(empty if none).\n\n"
        f"{rel_line(a['name'], b['name'], state.get('rel_a'))}"
        f"{a['name']} remembers:\n{_numbered(mem_a, names)}\n\n"
        f"{rel_line(b['name'], a['name'], state.get('rel_b'))}"
        f"{b['name']} remembers:\n{_numbered(mem_b, names)}\n\n"
        f"It is {clock_text(state.get('clock'))}" + (f", at {place}" if place else "") + "."
    )


def _feels(raw: Any) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    fact = re.sub(r"\s+", " ", str(raw.get("fact") or "")).strip()[:80]
    return {"affinity": _clamp_int(raw.get("affinity"), -3, 3, 0), "fact": fact}


def _sourced(raw: Any, notes: list[dict], heard: list[dict]) -> list[dict]:
    """Attach the speaker's memory id a note repeats, so the listener can say who told them."""
    kept = [n for n in (raw if isinstance(raw, list) else []) if clean_notes([n])]
    out = []
    for note, src in zip(notes, kept):
        n = src.get("from") if isinstance(src, dict) else None
        if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= len(heard):
            note = {**note, "from": heard[n - 1]["id"]}
        out.append(note)
    return out


def validate_chat(
    raw: Any,
    a: dict,
    b: dict,
    mem_a: Optional[list[dict]] = None,
    mem_b: Optional[list[dict]] = None,
) -> dict:
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
    out = {
        "a": a["id"],
        "b": b["id"],
        "lines": lines[:6],
        "a_learns": clean_notes(raw.get("a_learns")),
        "b_learns": clean_notes(raw.get("b_learns")),
    }
    if mem_a is None or mem_b is None:
        return out
    out["a_learns"] = _sourced(raw.get("a_learns"), out["a_learns"], mem_b)
    out["b_learns"] = _sourced(raw.get("b_learns"), out["b_learns"], mem_a)
    out["a_feels"] = _feels(raw.get("a_feels"))
    out["b_feels"] = _feels(raw.get("b_feels"))
    return out


def resolve_world_report(
    doc: dict, node_outputs: Optional[dict] = None
) -> tuple[Optional[dict], list[str]]:
    node_outputs = node_outputs or {}
    warnings: list[str] = []
    picked = most_downstream(doc, node_outputs, WORLD_SOURCE_TYPES, "world")
    raw: Optional[dict] = None
    source: Optional[str] = None
    if picked is not None and isinstance(picked[1], dict):
        raw, source = picked[1], picked[0]["id"]
    else:
        for n in doc.get("nodes") or []:
            if n["type"] != "world.spec" or n.get("disabled"):
                continue
            ins = input_values(doc, node_outputs, n["id"], "world")
            base = (
                ins[0] if ins and isinstance(ins[0], dict) else (n.get("params") or {}).get("world")
            )
            if isinstance(base, dict):
                raw, source = dict(base), n["id"]
                for ts in input_values(doc, node_outputs, n["id"], "tileset"):
                    if isinstance(ts, dict):
                        raw["tileset"] = {**(raw.get("tileset") or {}), **ts}
                break
    if raw is None:
        return None, ["This flow has no world yet: add a World node, or run a World Generator."]
    raw = dict(raw)
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
    world = WorldSpec.model_validate(apply_play_view(world, doc, node_outputs, source)).model_dump(
        exclude_none = True
    )
    reply = next((n for n in doc.get("nodes") or [] if n["id"] == eps.get("npc_reply")), None)
    if reply is not None and (reply.get("params") or {}).get("summarise") is False:
        world["summarise"] = False
    if reply is not None and (reply.get("params") or {}).get("new_quests") is False:
        world["new_quests"] = False
    level = (reply.get("params") or {}).get("autonomy") if reply is not None else None
    if level in AUTONOMY_LEVELS and level != "off":
        world["autonomy"] = {k: k in AUTONOMY_LEVELS[level] for k in AUTONOMY_FLAGS}
    if world.get("city") and raw.get("rivals"):
        warnings.append("This world turns on both city and rivals: only the Mayor panel shows")
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
