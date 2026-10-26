# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Describe a place: the LLM writes a symbolic scene (areas, props, relations) and a solver places it on the tiles.

After Holodeck (Yang et al. 2024): the model never picks coordinates. The solver puts each prop on a free tile the
player can reach, never lets a solid prop cut off open ground (BFS from spawn), and when a prop's relations cannot all
hold it drops them in priority order: ``facing``, then ``near``, then ``against_wall``, then ``in_area``.
"""

from __future__ import annotations

import math
import random
import re
from collections import deque
from typing import Any, Optional

from core.flows.authoring import AuthoringError, strict_schema, string_enum
from core.flows.story import slug
from core.flows.world import OPEN_PROPS, PROP_KINDS, props3d

KINDS = tuple(k for k in PROP_KINDS if k != "model")
RELATIONS = ("near", "against_wall", "facing", "in_area", "before_gate")
RELAX_ORDER = ("facing", "near", "against_wall", "in_area")
AREA_SIZES = {"small": 1, "medium": 2, "large": 3}
MAX_AREAS = 4
MAX_GROUPS = 12
MAX_COUNT = 8
MAX_PLACED = 40
NEAR = 2
FACING_RANGE = 6
WINDOW = (
    20  # tiles around the focus the solver looks at, so a big map costs no more than a small one
)
_WALL_PROPS = {"house", "fence", "stall"}
_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))

SCENE_SCHEMA = strict_schema(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "areas": {
                "type": "array",
                "maxItems": MAX_AREAS,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "size": string_enum(AREA_SIZES),
                    },
                    "required": ["id", "size"],
                },
            },
            "props": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_GROUPS,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "kind": string_enum(KINDS),
                        "count": {"type": "integer", "minimum": 1, "maximum": MAX_COUNT},
                        "area": {"type": "string"},
                    },
                    "required": ["id", "kind", "count", "area"],
                },
            },
            "relations": {
                "type": "array",
                "maxItems": 24,
                "items": {
                    "type": "object",
                    "properties": {
                        "type": string_enum(RELATIONS),
                        "a": {"type": "string"},
                        "b": {"type": "string"},
                    },
                    "required": ["type", "a", "b"],
                },
            },
        },
        "required": ["name", "areas", "props", "relations"],
    }
)

SCENE_SYSTEM = (
    "You lay out a small place in a cozy low-poly 3D village. Do not give coordinates. Answer in JSON with: `name`; "
    "`areas` (up to 4, each an `id` and a `size` small, medium or large); `props`, groups of scenery each with an "
    "`id`, a `kind` ("
    + ", ".join(KINDS)
    + "), a `count` from 1 to 8 and the `area` id it stands in; and "
    "`relations` between them: `near` (prop a close to prop b), `against_wall` (prop a backs onto a wall or "
    "building, b empty), `facing` (prop a turned towards prop b), `in_area` (prop a inside area b), `before_gate` "
    "(prop a reachable without passing a locked gate, b empty; for keys and levers). Keep it to what "
    "the description needs, at most 40 props in all."
)


def validate_scene(raw: Any) -> dict:
    """Ids unique, every reference resolves; areas default to one when the model gave none."""
    if not isinstance(raw, dict):
        raise AuthoringError("the answer is not a scene object")
    areas: list[dict] = []
    for a in raw.get("areas") or []:
        aid = slug(str(a.get("id") or ""), "area")
        if aid in {x["id"] for x in areas}:
            raise AuthoringError(f"area id {aid!r} is used twice")
        areas.append(
            {"id": aid, "size": a.get("size") if a.get("size") in AREA_SIZES else "medium"}
        )
    if not areas:
        areas = [{"id": "main", "size": "medium"}]
    area_ids = {a["id"] for a in areas}
    props: list[dict] = []
    total = 0
    for p in raw.get("props") or []:
        pid = slug(str(p.get("id") or ""), "prop")
        if pid in {x["id"] for x in props} or pid in area_ids:
            raise AuthoringError(f"prop id {pid!r} is used twice")
        if p.get("kind") not in KINDS:
            raise AuthoringError(
                f"prop {pid!r} has kind {p.get('kind')!r}, not one of {', '.join(KINDS)}"
            )
        area = slug(str(p.get("area") or ""), "") if p.get("area") else ""
        if area and area not in area_ids:
            raise AuthoringError(
                f"prop {pid!r} stands in area {p['area']!r}, which is not in `areas`"
            )
        count = max(1, min(MAX_COUNT, int(p.get("count") or 1)))
        total += count
        props.append({"id": pid, "kind": p["kind"], "count": count, "area": area or areas[0]["id"]})
    if not props:
        raise AuthoringError("the scene has no props")
    if total > MAX_PLACED:
        raise AuthoringError(f"the scene asks for {total} props; keep it to {MAX_PLACED}")
    prop_ids = {p["id"] for p in props}
    relations: list[dict] = []
    for r in raw.get("relations") or []:
        kind, a = r.get("type"), slug(str(r.get("a") or ""), "")
        b = slug(str(r.get("b") or ""), "") if r.get("b") else ""
        if kind not in RELATIONS:
            raise AuthoringError(f"relation {kind!r} is not one of {', '.join(RELATIONS)}")
        if a not in prop_ids:
            raise AuthoringError(
                f"relation {kind} names prop {r.get('a')!r}, which is not in `props`"
            )
        if kind in ("near", "facing") and b not in prop_ids:
            raise AuthoringError(f"relation {kind} needs `b` to be a prop id, got {r.get('b')!r}")
        if kind == "in_area" and b not in area_ids:
            raise AuthoringError(f"relation in_area needs `b` to be an area id, got {r.get('b')!r}")
        if kind in ("near", "facing") and a == b:
            continue
        relations.append(
            {"type": kind, "a": a, "b": b if kind not in ("against_wall", "before_gate") else ""}
        )
    return {
        "name": str(raw.get("name") or "A place")[:80],
        "areas": areas,
        "props": props,
        "relations": relations,
    }


_THEMES = (
    (
        ("market", "bazaar", "shop", "trade", "fair"),
        "Market",
        [("stall", 4), ("crate", 4), ("lamp", 2)],
    ),
    (
        ("forest", "grove", "wood", "glade", "tree"),
        "Grove",
        [("tree", 8), ("rock", 3), ("bench", 1)],
    ),
    (
        ("camp", "bandit", "raid", "hideout"),
        "Camp",
        [("crate", 5), ("fence", 4), ("lamp", 2), ("rock", 2)],
    ),
    (
        ("farm", "field", "barn", "pasture"),
        "Farmstead",
        [("house", 1), ("fence", 6), ("crate", 3), ("well", 1)],
    ),
    (("park", "garden", "rest", "quiet"), "Garden", [("tree", 4), ("bench", 3), ("lamp", 2)]),
    (("street", "town", "village", "houses"), "Lane", [("house", 3), ("lamp", 3), ("tree", 2)]),
)
_DEFAULT_THEME = ("Square", [("well", 1), ("bench", 3), ("lamp", 3), ("tree", 2)])


def fallback_scene(text: str) -> dict:
    """A procedural scene for when the model's never fits: picked by keywords in the description."""
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    name, groups = next(((n, g) for keys, n, g in _THEMES if words & set(keys)), _DEFAULT_THEME)
    props = [{"id": f"{k}s", "kind": k, "count": c, "area": "main"} for k, c in groups]
    relations = [{"type": "in_area", "a": p["id"], "b": "main"} for p in props]
    for p in props:
        if p["kind"] in ("bench", "stall"):
            relations.append({"type": "facing", "a": p["id"], "b": props[0]["id"]})
        if p["kind"] in ("crate", "lamp"):
            relations.append({"type": "near", "a": p["id"], "b": props[0]["id"]})
        if p["kind"] in ("house", "fence"):
            relations.append({"type": "against_wall", "a": p["id"], "b": ""})
    return validate_scene(
        {
            "name": name,
            "areas": [{"id": "main", "size": "large"}],
            "props": props,
            "relations": relations,
        }
    )


class Grid:
    """The walkable map: rows of ``#`` (solid), ``.`` (open), ``R`` (open, but kept clear: spawn, NPCs, items) and
    ``D`` (a locked gate: kept clear, walkable once unlocked, a wall for what must be reachable before it)."""

    def __init__(
        self,
        rows: list[str],
        spawn: Optional[tuple[int, int]] = None,
    ):
        self.h = len(rows)
        self.w = max((len(r) for r in rows), default = 0)
        if not self.w or not self.h:
            raise AuthoringError("the map is empty")
        self.solid = [[(r[x] if x < len(r) else "#") == "#" for x in range(self.w)] for r in rows]
        self.gates = {(x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c == "D"}
        self.reserved = {
            (x, y) for y, r in enumerate(rows) for x, c in enumerate(r) if c == "R"
        } | self.gates
        self.wall = [row[:] for row in self.solid]
        if spawn is None or not self.open(*spawn):
            spawn = next(
                ((x, y) for y in range(self.h) for x in range(self.w) if self.open(x, y)), None
            )
        if spawn is None:
            raise AuthoringError("the map has no open tile")
        self.spawn = spawn

    @classmethod
    def from_request(cls, req: dict) -> "Grid":
        rows = req.get("rows")
        if isinstance(rows, list) and rows and all(isinstance(r, str) for r in rows):
            sp = req.get("spawn") if isinstance(req.get("spawn"), dict) else {}
            spawn = (
                (int(sp["x"]), int(sp["y"]))
                if isinstance(sp.get("x"), int) and isinstance(sp.get("y"), int)
                else None
            )
            return cls([r[:256] for r in rows[:256]], spawn)
        w = max(8, min(256, int(req.get("width") or 24)))
        h = max(8, min(256, int(req.get("height") or 18)))
        rows = ["#" * w] + ["#" + "." * (w - 2) + "#" for _ in range(h - 2)] + ["#" * w]
        return cls(rows, (w // 2, h // 2))

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def open(self, x: int, y: int) -> bool:
        return self.inside(x, y) and not self.solid[y][x]

    def reach(self, locked: bool = False) -> set[tuple[int, int]]:
        """Tiles walkable from spawn; ``locked`` treats gates as walls."""
        seen = {self.spawn}
        todo = deque([self.spawn])
        while todo:
            x, y = todo.popleft()
            for dx, dy in _STEPS:
                n = (x + dx, y + dy)
                if n not in seen and self.open(*n) and not (locked and n in self.gates):
                    seen.add(n)
                    todo.append(n)
        return seen


def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _order(props: list[dict], relations: list[dict]) -> list[dict]:
    """Targets of `near` / `facing` first, so they are on the map when their partners look for them."""
    deps = {
        p["id"]: {
            r["b"] for r in relations if r["a"] == p["id"] and r["type"] in ("near", "facing")
        }
        for p in props
    }
    out: list[dict] = []
    done: set[str] = set()
    pending = list(props)
    while pending:
        ready = [p for p in pending if deps[p["id"]] <= done] or pending[:1]
        p = ready[0]
        out.append(p)
        done.add(p["id"])
        pending.remove(p)
    return out


def solve(
    spec: dict,
    grid: Grid,
    *,
    focus: Optional[tuple[int, int]] = None,
    seed: int = 0,
    taken_ids: Optional[set[str]] = None,
) -> dict:
    """Place ``spec`` (from ``validate_scene``) on ``grid``. Never fails: what cannot fit is reported in ``dropped``."""
    rng = random.Random(seed)
    reach = grid.reach()
    before = grid.reach(locked = True)
    free = {c for c in reach if c not in grid.reserved and c != grid.spawn}
    if focus is None or focus not in reach:
        f = focus or grid.spawn
        focus = min(reach, key = lambda c: (_cheb(c, f), c[1], c[0]))
    free = {c for c in free if _cheb(c, focus) <= WINDOW}

    areas: dict[str, dict] = {}
    for a in spec["areas"]:
        r = AREA_SIZES[a["size"]] + 1
        if not areas:
            centre = focus
        else:

            def score(c: tuple[int, int]) -> tuple:
                gap = min(_cheb(c, b["centre"]) - b["r"] - r for b in areas.values())
                room = sum(
                    1
                    for dx in range(-r, r + 1)
                    for dy in range(-r, r + 1)
                    if (c[0] + dx, c[1] + dy) in free
                )
                return (gap < 1, -room, _cheb(c, focus), c[1], c[0])

            near_focus = [c for c in free if _cheb(c, focus) <= 3 * (r + 2) + 2 * len(areas)]
            centre = min(near_focus or free or [focus], key = score)
        areas[a["id"]] = {"id": a["id"], "centre": centre, "r": r}

    placed: dict[str, list[tuple[int, int]]] = {}
    occupied: set[tuple[int, int]] = set()
    solid_spots: list[tuple[int, int]] = []
    key_spots: set[tuple[int, int]] = set()
    reach_count = len(reach)
    before_count = len(before)
    out: list[dict] = []
    relaxed: list[dict] = []
    dropped: list[str] = []
    ids = set(taken_ids or ())

    def wall_next(c: tuple[int, int]) -> Optional[tuple[int, int]]:
        for dx, dy in _STEPS:
            n = (c[0] + dx, c[1] + dy)
            if not grid.inside(*n) or grid.wall[n[1]][n[0]]:
                return (dx, dy)
        return None

    def holds(rel: dict, c: tuple[int, int], group: dict) -> bool:
        if rel["type"] == "in_area":
            a = areas[rel["b"]]
            return _cheb(c, a["centre"]) <= a["r"]
        if rel["type"] == "before_gate":
            return c in before
        if rel["type"] == "against_wall":
            return wall_next(c) is not None
        others = placed.get(rel["b"]) or []
        if not others:
            return True
        reach_d = NEAR if rel["type"] == "near" else FACING_RANGE
        return min(_cheb(c, o) for o in others) <= reach_d

    def touches(c: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
        return any((c[0] + dx, c[1] + dy) in cells for dx, dy in _STEPS)

    def keeps_reach(c: tuple[int, int]) -> bool:
        """A solid prop may not cut off open ground or wall in another solid prop the player could walk up to."""
        grid.solid[c[1]][c[0]] = True
        now = grid.reach()
        ok = len(now) == reach_count - 1 and all(touches(s, now) for s in (*solid_spots, c))
        if ok and grid.gates:
            locked = grid.reach(locked = True)
            ok = len(locked) == before_count - (c in before) and all(
                touches(s, locked) if s in key_spots else True for s in (*solid_spots, c)
            )
        if not ok:
            grid.solid[c[1]][c[0]] = False
        return ok

    for group in _order(spec["props"], spec["relations"]):
        rels = [r for r in spec["relations"] if r["a"] == group["id"]]
        if not any(r["type"] == "in_area" for r in rels):
            rels.append({"type": "in_area", "a": group["id"], "b": group["area"]})
        home = areas[group["area"]]["centre"]
        solid = group["kind"] not in OPEN_PROPS
        for n in range(group["count"]):
            active = list(rels)
            dropped_here: list[str] = []
            spot = None
            while spot is None:
                cands = [
                    c for c in free if c not in occupied and all(holds(r, c, group) for r in active)
                ]
                spread = placed.get(group["id"]) or []
                cands.sort(
                    key = lambda c: (
                        sum(1 for o in occupied if _cheb(c, o) <= 1)
                        if group["kind"] != "fence"
                        else 0,
                        -min((_cheb(c, o) for o in spread), default = 0)
                        if group["kind"] != "fence"
                        else 0,
                        _cheb(c, home),
                        rng.random(),
                    )
                )
                gated = any(r["type"] == "before_gate" for r in active)
                for c in cands:
                    if gated:
                        key_spots.add(c)
                    if not solid or keeps_reach(c):
                        spot = c
                        break
                    key_spots.discard(c)
                if spot is not None:
                    break
                droppable = next(
                    (t for t in RELAX_ORDER if any(r["type"] == t for r in active)), None
                )
                if droppable is None:
                    break
                active = [r for r in active if r["type"] != droppable]
                dropped_here.append(droppable)
            if spot is None:
                dropped.append(f"{group['id']}#{n + 1}")
                continue
            if solid:
                reach_count -= 1
                before_count -= spot in before
                solid_spots.append(spot)
                if group["kind"] in _WALL_PROPS:
                    grid.wall[spot[1]][spot[0]] = True
            if dropped_here:
                relaxed.append(
                    {
                        "prop": f"{group['id']}#{n + 1}",
                        "dropped": sorted(set(dropped_here), key = RELAX_ORDER.index),
                    }
                )
            occupied.add(spot)
            placed.setdefault(group["id"], []).append(spot)
            rot = 0.0
            face = next((r for r in active if r["type"] == "facing" and placed.get(r["b"])), None)
            if face is not None:
                t = min(placed[face["b"]], key = lambda o: _cheb(spot, o))
                rot = math.degrees(math.atan2(t[0] - spot[0], t[1] - spot[1])) % 360
            elif any(r["type"] == "against_wall" for r in active) and (d := wall_next(spot)):
                rot = math.degrees(math.atan2(-d[0], -d[1])) % 360
            k = 1
            while f"{group['kind']}-{k}" in ids:
                k += 1
            pid = f"{group['kind']}-{k}"
            ids.add(pid)
            out.append(
                {
                    "id": pid,
                    "kind": group["kind"],
                    "name": group["id"].replace("_", " ")[:60],
                    "x": spot[0] + 0.5,
                    "y": spot[1] + 0.5,
                    "rot": round(rot / 90) * 90 % 360
                    if group["kind"] in ("house", "fence", "stall")
                    else round(rot, 1),
                    "scale": 1,
                    "solid": solid,
                }
            )
    final = grid.reach()
    final_locked = grid.reach(locked = True)

    def walkable_to(p: dict, cells: set[tuple[int, int]]) -> bool:
        c = (int(p["x"]), int(p["y"]))
        return touches(c, cells) if p["solid"] else c in cells

    reachable = all(walkable_to(p, final) for p in out) and all(
        walkable_to(p, final_locked) for p in out if (int(p["x"]), int(p["y"])) in key_spots
    )
    return {
        "name": spec["name"],
        "props3d": props3d(out, grid.w, grid.h),
        "areas": [
            {"id": a["id"], "x": a["centre"][0], "y": a["centre"][1], "r": a["r"]}
            for a in areas.values()
        ],
        "relaxed": relaxed,
        "dropped": dropped,
        "reachable": reachable,
        "open_tiles_cut": len(reach) - len(final) - sum(1 for p in out if p["solid"]),
    }
