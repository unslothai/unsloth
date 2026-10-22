# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Story manifests: compile an authored Story/Scene/Choice graph, pick a generated story, repair LLM output.

Player conventions (see `studio/frontend/src/features/flows/player/story/reducer.ts`):
- A scene with no lines is a router: it applies `on_enter`, then leaves at once by `random`, by the first
  choice whose `if` holds when every choice text is empty, or by `next`.
- A scene with no way out and no `ending` is an implicit "The End"."""

from __future__ import annotations

import re
from typing import Any, Optional

from core.flows.expr import Prng, seed_from
from core.flows.schema import StoryManifest, StoryScene

STORY_SOURCE_TYPES = ("story.compile", "story.narrate", "story.illustrate", "story.generator")
STORY_PLAYER_TYPES = (
    "story.start",
    "story.scene",
    "story.choice",
    "story.branch",
    "story.random",
    "story.set",
    "story.ending",
)
CHARACTER_COLORS = (
    "#f472b6",
    "#60a5fa",
    "#fbbf24",
    "#34d399",
    "#c084fc",
    "#f87171",
    "#2dd4bf",
    "#fb923c",
)
MAX_HOPS = 200


class StoryCompileError(ValueError):
    pass


def slug(text: Any, fallback: str = "x") -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text or "").lower()).strip("_")[:40]
    return s or fallback


def _unique(base: str, taken: set[str]) -> str:
    out, i = base, 2
    while out in taken:
        out = f"{base}_{i}"
        i += 1
    taken.add(out)
    return out


# ---------------------------------------------------------------------------
# Graph helpers shared with world.py
# ---------------------------------------------------------------------------
def node_value(doc: dict, node_outputs: dict, node_id: str, port: str) -> Any:
    """A node's output: baked first (Appendix B), then the latest run."""
    baked = (doc.get("baked") or {}).get(node_id) or {}
    if port in baked and baked[port] is not None:
        return baked[port]
    return ((node_outputs or {}).get(node_id) or {}).get(port)


def input_values(doc: dict, node_outputs: dict, node_id: str, port: str) -> list[Any]:
    vals = []
    for e in doc.get("edges") or []:
        if e.get("kind", "data") == "data" and e["target"] == node_id and e["target_port"] == port:
            v = node_value(doc, node_outputs, e["source"], e["source_port"])
            if v is not None:
                vals.append(v)
    return vals


def data_depths(doc: dict) -> dict[str, int]:
    """Longest data-edge path length to every node (cycles are cut)."""
    preds: dict[str, list[str]] = {n["id"]: [] for n in doc.get("nodes") or []}
    for e in doc.get("edges") or []:
        if e.get("kind", "data") == "data" and e["target"] in preds and e["source"] in preds:
            preds[e["target"]].append(e["source"])
    depth: dict[str, int] = {}
    visiting: set[str] = set()

    def walk(n: str) -> int:
        if n in depth:
            return depth[n]
        if n in visiting:
            return 0
        visiting.add(n)
        d = max((walk(p) + 1 for p in preds.get(n, [])), default = 0)
        visiting.discard(n)
        depth[n] = d
        return d

    for n in preds:
        walk(n)
    return depth


def most_downstream(
    doc: dict, node_outputs: dict, types: tuple[str, ...], port: str
) -> Optional[tuple[dict, Any]]:
    depth = data_depths(doc)
    best: Optional[tuple[int, int, dict, Any]] = None
    for n in doc.get("nodes") or []:
        if n["type"] not in types or n.get("disabled"):
            continue
        v = node_value(doc, node_outputs, n["id"], port)
        if v is None:
            continue
        rank = (depth.get(n["id"], 0), -types.index(n["type"]))
        if best is None or rank > best[:2]:
            best = (rank[0], rank[1], n, v)
    return (best[2], best[3]) if best else None


def _entry_points(doc: dict, kinds: dict[str, str]) -> dict[str, str]:
    eps: dict[str, str] = {}
    for n in doc.get("nodes") or []:
        name = kinds.get(n["type"])
        if name and not n.get("disabled") and name not in eps:
            eps[name] = n["id"]
    return eps


STORY_ENTRY_TYPES = {"story.next_scene": "next_scene", "story.scene_art": "scene_art"}


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------
_LINE_RE = re.compile(r"^\s*([^:\n]{1,40}?)\s*(?:\(([^)]{1,30})\))?\s*:\s*(.+)$")


def parse_lines(
    text: str,
    characters: list[dict],
    *,
    auto_characters: bool = True,
) -> list[dict]:
    """`Name: text` / `Name (expression): text` lines; `Narrator:` or no prefix = narrator."""
    by_name = {c["name"].lower(): c for c in characters}
    by_id = {c["id"]: c for c in characters}
    taken = {c["id"] for c in characters}
    lines = []
    for raw in str(text or "").splitlines():
        if not raw.strip():
            continue
        m = _LINE_RE.match(raw)
        speaker, expression, body = None, None, raw.strip()
        if m:
            name = m.group(1).strip()
            key = name.lower()
            char = by_name.get(key) or by_id.get(key)
            plausible = len(name.split()) <= 3 and all(w[:1].isupper() for w in name.split())
            if key in ("narrator", "narration", "n"):
                body = m.group(3).strip()
            elif char is not None:
                speaker, expression, body = char["id"], m.group(2), m.group(3).strip()
            elif auto_characters and plausible:
                char = {
                    "id": _unique(slug(name, "character"), taken),
                    "name": name,
                    "color": CHARACTER_COLORS[len(characters) % len(CHARACTER_COLORS)],
                    "side": "right" if len(characters) % 2 else "left",
                }
                characters.append(char)
                by_name[key] = char
                by_id[char["id"]] = char
                speaker, expression, body = char["id"], m.group(2), m.group(3).strip()
        line: dict[str, Any] = {"speaker": speaker, "text": body}
        if expression:
            line["expression"] = expression.strip().lower()
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Authored graph -> manifest
# ---------------------------------------------------------------------------
def _flow_successors(doc: dict) -> dict[tuple[str, str], str]:
    succ: dict[tuple[str, str], str] = {}
    for e in doc.get("edges") or []:
        if e.get("kind") == "flow":
            succ.setdefault((e["source"], e["source_port"]), e["target"])
    return succ


def _index_key(d: Any, i: int) -> Any:
    """Choice conditions/effects keyed by 0-based index ("0") or given as a list."""
    if isinstance(d, dict):
        return d.get(str(i))
    if isinstance(d, list) and i < len(d):
        return d[i]
    return None


def compile_authored(doc: dict, node_outputs: dict, warnings: list[str]) -> Optional[dict]:
    nodes = {n["id"]: n for n in doc.get("nodes") or [] if not n.get("disabled")}
    if not any(
        n["type"] in STORY_PLAYER_TYPES and n["type"] != "story.start" for n in nodes.values()
    ):
        return None
    succ = _flow_successors(doc)
    start_node = next((n for n in nodes.values() if n["type"] == "story.start"), None)
    sp = (start_node or {}).get("params") or {}
    infinite = sp.get("mode") == "infinite"

    characters: list[dict] = []
    taken: set[str] = set()
    char_by_node: dict[str, str] = {}
    for n in sorted(
        nodes.values(), key = lambda n: (n["pos"][0], n["pos"][1]) if n.get("pos") else (0, 0)
    ):
        if n["type"] != "story.character":
            continue
        p = n.get("params") or {}
        name = str(p.get("name") or "Character").strip()
        cid = _unique(slug(name, "character"), taken)
        char_by_node[n["id"]] = cid
        portraits = input_values(doc, node_outputs, n["id"], "portrait")
        characters.append(
            {
                "id": cid,
                "name": name,
                "color": p.get("color")
                or CHARACTER_COLORS[len(characters) % len(CHARACTER_COLORS)],
                "look": p.get("look") or None,
                "voice": p.get("voice") or None,
                "side": p.get("side") or "left",
                "portrait": portraits[0] if portraits else None,
            }
        )

    scenes: list[dict] = []

    def target(node_id: str, handle: str) -> Optional[str]:
        t = succ.get((node_id, handle))
        if t is None:
            return None
        if (
            t not in nodes
            or nodes[t]["type"] not in STORY_PLAYER_TYPES
            or nodes[t]["type"] == "story.start"
        ):
            warnings.append(f"{node_id}.{handle} leads to a node that is not a scene")
            return None
        return t

    for nid, n in nodes.items():
        t, p = n["type"], n.get("params") or {}
        if t == "story.scene":
            text = (
                "\n".join(str(v) for v in input_values(doc, node_outputs, nid, "lines_text"))
                or p.get("lines")
                or ""
            )
            lines = parse_lines(text, characters)
            on_stage: list[str] = []
            for v in input_values(doc, node_outputs, nid, "characters"):
                for c in v if isinstance(v, list) else [v]:
                    if isinstance(c, dict) and c.get("name"):
                        hit = next(
                            (
                                x["id"]
                                for x in characters
                                if x["name"].lower() == str(c["name"]).lower()
                            ),
                            None,
                        )
                        if hit and hit not in on_stage:
                            on_stage.append(hit)
            for e in doc.get("edges") or []:
                if (
                    e["target"] == nid
                    and e["target_port"] == "characters"
                    and e["source"] in char_by_node
                ):
                    if char_by_node[e["source"]] not in on_stage:
                        on_stage.append(char_by_node[e["source"]])
            for ln in lines:
                if ln["speaker"] and ln["speaker"] not in on_stage:
                    on_stage.append(ln["speaker"])
            bg = input_values(doc, node_outputs, nid, "background")
            music = input_values(doc, node_outputs, nid, "music")
            scenes.append(
                {
                    "id": nid,
                    "title": p.get("title") or None,
                    "background": bg[0] if bg else None,
                    "background_prompt": p.get("background_prompt") or None,
                    "music": music[0] if music else None,
                    "on_stage": on_stage,
                    "lines": lines,
                    "choices": [],
                    "next": target(nid, "next"),
                    "on_enter": p.get("on_enter") or {},
                    "generate_background": bool(p.get("generate_background")),
                    "generate_lines": bool(p.get("generate_lines")),
                }
            )
        elif t == "story.choice":
            choices = []
            for i, text in enumerate((p.get("choices") or [])[:4]):
                goto = target(nid, f"choice_{i + 1}")
                if goto is None and infinite:
                    # Infinite mode: an open exit asks the Next Scene entry point to write what follows.
                    goto = f"gen_{nid}_{i + 1}"
                elif goto is None:
                    warnings.append(f"choice {text!r} of {nid} leads nowhere and was dropped")
                    continue
                c: dict[str, Any] = {"id": f"{nid}_{i + 1}", "text": str(text), "goto": goto}
                cond = _index_key(p.get("conditions"), i)
                if cond is not None:
                    c["if"] = cond
                eff = _index_key(p.get("effects"), i)
                if isinstance(eff, dict):
                    c["set"] = eff
                choices.append(c)
            prompt = str(p.get("prompt") or "").strip()
            scenes.append(
                {
                    "id": nid,
                    "lines": [{"speaker": None, "text": prompt}] if prompt else [],
                    "choices": choices,
                    "_choice": True,
                }
            )
        elif t == "story.branch":
            then, other = target(nid, "then"), target(nid, "else")
            choices = []
            if then:
                choices.append(
                    {"id": f"{nid}_then", "text": "", "goto": then, "if": p.get("condition")}
                )
            if other:
                choices.append({"id": f"{nid}_else", "text": "", "goto": other})
            scenes.append({"id": nid, "lines": [], "choices": choices})
        elif t == "story.random":
            weights = p.get("weights") if isinstance(p.get("weights"), list) else []
            rnd = []
            for i in range(3):
                goto = target(nid, f"path_{i + 1}")
                if goto:
                    w = (
                        weights[i]
                        if i < len(weights) and isinstance(weights[i], (int, float))
                        else 1
                    )
                    rnd.append({"weight": max(float(w), 0.0), "goto": goto})
            scenes.append({"id": nid, "lines": [], "choices": [], "random": rnd})
        elif t == "story.set":
            scenes.append(
                {
                    "id": nid,
                    "lines": [],
                    "choices": [],
                    "on_enter": p.get("set") or {},
                    "next": target(nid, "next"),
                }
            )
        elif t == "story.ending":
            text = str(p.get("text") or "").strip()
            scenes.append(
                {
                    "id": nid,
                    "title": p.get("title") or "The End",
                    "lines": parse_lines(text, characters, auto_characters = False) if text else [],
                    "choices": [],
                    "ending": {
                        "title": p.get("title") or "The End",
                        "kind": p.get("kind") or "neutral",
                    },
                }
            )

    # A scene that flows straight into a Choice shows the menu itself, under its own background.
    by_id = {s["id"]: s for s in scenes}
    for s in scenes:
        nxt = by_id.get(s.get("next") or "")
        if nxt is not None and nxt.get("_choice") and not s.get("_choice") and not s["choices"]:
            s["choices"] = [dict(c) for c in nxt["choices"]]
            s["lines"] = s["lines"] + nxt["lines"]
            s["next"] = None
            nxt["_merged"] = True
    targeted = {t for s in scenes if not s.get("_merged") for t in _targets(s)}
    scenes = [
        s for s in scenes if not (s.get("_choice") and s.get("_merged") and s["id"] not in targeted)
    ]
    for s in scenes:
        s.pop("_choice", None)
        s.pop("_merged", None)

    start = target(start_node["id"], "next") if start_node else None
    if start is None:
        first = sorted(
            (n for n in nodes.values() if n["type"] == "story.scene"),
            key = lambda n: tuple(n.get("pos") or (0, 0)),
        )
        if not first:
            raise StoryCompileError("the story has no scenes")
        start = first[0]["id"]
        if start_node:
            warnings.append(
                "Story is not connected to a first Scene; starting at the leftmost scene"
            )

    vars_ = sp.get("vars") if isinstance(sp.get("vars"), dict) else {}
    return {
        "title": sp.get("title") or doc.get("name") or "Untitled Story",
        "start": start,
        "vars": vars_,
        "characters": characters,
        "scenes": scenes,
        "mode": sp.get("mode") or "fixed",
        "style": sp.get("style") or None,
    }


# ---------------------------------------------------------------------------
# Validation and repair (authored and generated)
# ---------------------------------------------------------------------------
def _reaches_ending(scenes: dict[str, dict]) -> set[str]:
    rev: dict[str, set[str]] = {sid: set() for sid in scenes}
    for sid, s in scenes.items():
        for t in _targets(s):
            if t in rev:
                rev[t].add(sid)
    good = {sid for sid, s in scenes.items() if s.get("ending") or not _targets(s)}
    stack = list(good)
    while stack:
        cur = stack.pop()
        for p in rev.get(cur, ()):
            if p not in good:
                good.add(p)
                stack.append(p)
    return good


def _targets(s: dict) -> list[str]:
    out = [c["goto"] for c in s.get("choices") or []]
    out += [r["goto"] for r in s.get("random") or []]
    if s.get("next"):
        out.append(s["next"])
    return out


def finalize(
    manifest: dict,
    warnings: list[str],
    *,
    allow_unknown_gotos: bool = False,
) -> dict:
    """Drop dangling transitions, guarantee a start, and validate against StoryManifest."""
    scenes = {s["id"]: s for s in manifest.get("scenes") or []}
    if not scenes:
        raise StoryCompileError("the story has no scenes")
    for s in scenes.values():
        keep = []
        for c in s.get("choices") or []:
            if c.get("goto") in scenes or allow_unknown_gotos:
                keep.append(c)
            else:
                warnings.append(f"choice {c.get('text')!r} in {s['id']} points to a missing scene")
        s["choices"] = keep
        s["random"] = [r for r in s.get("random") or [] if r.get("goto") in scenes]
        if s.get("next") and s["next"] not in scenes and not allow_unknown_gotos:
            warnings.append(f"{s['id']} continues to a missing scene")
            s["next"] = None
    if manifest.get("start") not in scenes:
        manifest["start"] = next(iter(scenes))
    char_ids = {c["id"] for c in manifest.get("characters") or []}
    for s in scenes.values():
        s["on_stage"] = [c for c in s.get("on_stage") or [] if c in char_ids]
        for ln in s.get("lines") or []:
            if ln.get("speaker") and ln["speaker"] not in char_ids:
                ln["speaker"] = None
    manifest["scenes"] = list(scenes.values())
    return StoryManifest.model_validate(manifest).model_dump(by_alias = True, exclude_none = True)


def repair_generated(
    raw: Any,
    *,
    endings: int = 3,
    seed: int = 0,
    style: Optional[str] = None,
    mode: str = "live",
) -> dict:
    """Turn an LLM's story JSON (the `STORY_SCHEMA` shape, loosely) into a valid, finishable manifest."""
    if not isinstance(raw, dict):
        raise StoryCompileError("the model did not return a story object")
    rng = Prng(seed_from("repair", seed))
    taken_c: set[str] = set()
    characters = []
    char_lookup: dict[str, str] = {}
    for i, c in enumerate(raw.get("characters") or []):
        if not isinstance(c, dict) or not c.get("name"):
            continue
        cid = _unique(slug(c.get("id") or c["name"], "character"), taken_c)
        for key in (str(c.get("id") or "").lower(), str(c["name"]).lower(), cid):
            if key:
                char_lookup[key] = cid
        side = (
            c.get("side")
            if c.get("side") in ("left", "right", "center")
            else ("left" if i % 2 == 0 else "right")
        )
        characters.append(
            {
                "id": cid,
                "name": str(c["name"])[:60],
                "look": str(c.get("look") or "")[:400] or None,
                "color": c.get("color")
                if re.fullmatch(r"#[0-9a-fA-F]{6}", str(c.get("color") or ""))
                else CHARACTER_COLORS[i % len(CHARACTER_COLORS)],
                "voice": c.get("voice") or None,
                "side": side,
            }
        )

    taken_s: set[str] = set()
    scene_lookup: dict[str, str] = {}
    raw_scenes = [s for s in raw.get("scenes") or [] if isinstance(s, dict)]
    for s in raw_scenes:
        sid = _unique(slug(s.get("id") or s.get("title"), "scene"), taken_s)
        for key in (str(s.get("id") or "").lower(), slug(s.get("id")), slug(s.get("title")), sid):
            if key and key not in scene_lookup:
                scene_lookup[key] = sid
        s["_id"] = sid
    if not raw_scenes:
        raise StoryCompileError("the model wrote no scenes")

    def resolve(goto: Any) -> Optional[str]:
        key = str(goto or "").strip().lower()
        return scene_lookup.get(key) or scene_lookup.get(slug(key))

    vars_: dict[str, Any] = {}
    for v in raw.get("vars") or []:
        if isinstance(v, dict) and v.get("name"):
            val = v.get("value", 0)
            vars_[slug(v["name"])] = (
                val if isinstance(val, (int, float)) and not isinstance(val, bool) else 0
            )

    scenes = []
    for s in raw_scenes:
        lines = []
        for ln in s.get("lines") or []:
            if not isinstance(ln, dict) or not str(ln.get("text") or "").strip():
                continue
            spk = str(ln.get("speaker") or "").strip().lower()
            lines.append(
                {
                    "speaker": char_lookup.get(spk) or char_lookup.get(slug(spk)),
                    "text": str(ln["text"]).strip()[:1200],
                }
            )
        choices = []
        for j, c in enumerate(s.get("choices") or []):
            if not isinstance(c, dict) or not str(c.get("text") or "").strip():
                continue
            goto = resolve(c.get("goto"))
            if goto is None or goto == s["_id"]:
                continue
            choice: dict[str, Any] = {
                "id": f"{s['_id']}_{j + 1}",
                "text": str(c["text"]).strip()[:200],
                "goto": goto,
            }
            sets = {}
            for eff in c.get("effects") or []:
                if isinstance(eff, dict) and eff.get("var"):
                    name = slug(eff["var"])
                    vars_.setdefault(name, 0)
                    add = eff.get("add", 1)
                    add = add if isinstance(add, (int, float)) and not isinstance(add, bool) else 1
                    sets[name] = {"op": "+", "args": [{"var": name}, add]}
            if sets:
                choice["set"] = sets
            req = slug(c.get("requires_var")) if c.get("requires_var") else None
            if req and req != "x":
                vars_.setdefault(req, 0)
                choice["if"] = {
                    "op": ">=",
                    "args": [
                        {"var": req},
                        c.get("requires_min")
                        if isinstance(c.get("requires_min"), (int, float))
                        else 1,
                    ],
                }
            choices.append(choice)
        on_stage = []
        for cid in s.get("on_stage") or []:
            hit = char_lookup.get(str(cid).lower()) or char_lookup.get(slug(cid))
            if hit and hit not in on_stage:
                on_stage.append(hit)
        for ln in lines:
            if ln["speaker"] and ln["speaker"] not in on_stage:
                on_stage.append(ln["speaker"])
        ending = s.get("ending") if isinstance(s.get("ending"), dict) else {}
        is_end = bool(s.get("is_ending")) or bool(ending.get("title"))
        scene: dict[str, Any] = {
            "id": s["_id"],
            "title": str(s.get("title") or "")[:120] or None,
            "background_prompt": str(s.get("background_prompt") or s.get("title") or "")[:600]
            or None,
            "on_stage": on_stage,
            "lines": lines or [{"speaker": None, "text": str(s.get("title") or "...")}],
            "choices": [] if is_end else choices,
        }
        if not is_end and not choices:
            nxt = resolve(s.get("next"))
            if nxt and nxt != s["_id"]:
                scene["next"] = nxt
        if is_end:
            kind = str(s.get("ending_kind") or ending.get("kind") or "neutral")
            title = s.get("ending_title") or ending.get("title")
            scene["ending"] = {
                "title": str(title or s.get("title") or "The End")[:120],
                "kind": kind if kind in ("good", "bad", "neutral", "secret") else "neutral",
            }
        scenes.append(scene)

    by_id = {s["id"]: s for s in scenes}
    ending_ids = [s["id"] for s in scenes if s.get("ending")]
    if not ending_ids:
        # The last scene ends the story.
        last = scenes[-1]
        last["choices"] = []
        last.pop("next", None)
        last["ending"] = {"title": last.get("title") or "The End", "kind": "neutral"}
        ending_ids = [last["id"]]

    # Dead ends that are not endings continue to the next scene in writing order, or an ending.
    order = [s["id"] for s in scenes]
    for i, s in enumerate(scenes):
        if not s.get("ending") and not s["choices"] and not s.get("next"):
            later = [x for x in order[i + 1 :] if x != s["id"]]
            s["next"] = later[0] if later else ending_ids[rng.randint(0, len(ending_ids) - 1)]

    # Every scene must be able to reach an ending (no closed loops).
    good = _reaches_ending(by_id)
    for s in scenes:
        if s["id"] not in good:
            goto = ending_ids[rng.randint(0, len(ending_ids) - 1)]
            if s["choices"]:
                s["choices"].append(
                    {"id": f"{s['id']}_out", "text": "Find another way", "goto": goto}
                )
            else:
                s["next"] = goto
    good = _reaches_ending(by_id)
    assert all(s["id"] in good for s in scenes)

    start = resolve(raw.get("start")) or scenes[0]["id"]
    manifest = {
        "title": str(raw.get("title") or "Untitled Story")[:200],
        "start": start,
        "vars": vars_,
        "characters": characters,
        "scenes": scenes,
        "mode": mode if mode in ("fixed", "live", "infinite") else "live",
        "style": style or raw.get("style") or None,
    }
    return finalize(manifest, [])


def story_schema(
    scenes: int,
    characters: int,
    choices_max: int = 3,
) -> dict:
    """The JSON schema `story.generator` asks the model to fill (kept flat for grammar-constrained decoding)."""
    line = {
        "type": "object",
        "properties": {"speaker": {"type": "string"}, "text": {"type": "string"}},
        "required": ["speaker", "text"],
    }
    effect = {
        "type": "object",
        "properties": {"var": {"type": "string"}, "add": {"type": "integer"}},
        "required": ["var", "add"],
    }
    choice = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "goto": {"type": "string"},
            "effects": {"type": "array", "items": effect, "maxItems": 2},
        },
        "required": ["text", "goto"],
    }
    scene = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "background_prompt": {"type": "string"},
            "on_stage": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "lines": {"type": "array", "items": line, "minItems": 1, "maxItems": 8},
            "choices": {"type": "array", "items": choice, "maxItems": choices_max},
            "is_ending": {"type": "boolean"},
            "ending_title": {"type": "string"},
            "ending_kind": {"type": "string", "enum": ["good", "bad", "neutral", "secret"]},
        },
        "required": ["id", "title", "background_prompt", "lines", "choices", "is_ending"],
    }
    character = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "look": {"type": "string"},
            "side": {"type": "string", "enum": ["left", "right"]},
        },
        "required": ["id", "name", "look"],
    }
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "start": {"type": "string"},
            "vars": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "value": {"type": "integer"}},
                    "required": ["name", "value"],
                },
            },
            "characters": {
                "type": "array",
                "items": character,
                "minItems": 1,
                "maxItems": max(1, characters),
            },
            "scenes": {"type": "array", "items": scene, "minItems": 2, "maxItems": max(2, scenes)},
        },
        "required": ["title", "start", "characters", "scenes"],
    }


def scene_schema(choices: int) -> dict:
    s = story_schema(1, 1, choices)["properties"]["scenes"]["items"]
    s = {**s, "properties": dict(s["properties"])}
    s["properties"]["choices"] = {
        "type": "array",
        "maxItems": choices,
        "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "effects": s["properties"]["choices"]["items"]["properties"]["effects"],
            },
            "required": ["text"],
        },
    }
    s["required"] = ["title", "background_prompt", "lines", "choices", "is_ending"]
    return s


def repair_live_scene(
    raw: Any,
    state: dict,
    *,
    choices: int = 3,
) -> dict:
    """An LLM-written next scene -> a valid StoryScene whose choices lead to new scene ids (`gen_*`),
    or to existing scenes when the model names one."""
    if not isinstance(raw, dict):
        raise StoryCompileError("the model did not return a scene object")
    story = state.get("story") or {}
    chars = story.get("characters") or []
    char_lookup = {}
    for c in chars:
        char_lookup[str(c.get("id", "")).lower()] = c.get("id")
        char_lookup[str(c.get("name", "")).lower()] = c.get("id")
    known = {s.get("id") for s in story.get("scenes") or []}
    depth = len(state.get("history") or [])
    base = seed_from("live", state.get("scene_id"), depth, state.get("choice"))
    sid = f"gen_{base:08x}"
    lines = []
    for ln in raw.get("lines") or []:
        if isinstance(ln, dict) and str(ln.get("text") or "").strip():
            spk = str(ln.get("speaker") or "").strip().lower()
            lines.append({"speaker": char_lookup.get(spk), "text": str(ln["text"]).strip()[:1200]})
    scene: dict[str, Any] = {
        "id": sid,
        "title": str(raw.get("title") or "")[:120] or None,
        "background_prompt": str(raw.get("background_prompt") or raw.get("title") or "")[:600]
        or None,
        "lines": lines or [{"speaker": None, "text": "..."}],
        "choices": [],
        "on_stage": [x for x in dict.fromkeys(ln["speaker"] for ln in lines if ln["speaker"])],
        "generate_background": True,
    }
    if raw.get("is_ending"):
        kind = (
            raw.get("ending_kind")
            if raw.get("ending_kind") in ("good", "bad", "neutral", "secret")
            else "neutral"
        )
        scene["ending"] = {
            "title": str(raw.get("ending_title") or raw.get("title") or "The End")[:120],
            "kind": kind,
        }
    else:
        for j, c in enumerate((raw.get("choices") or [])[:choices]):
            if not isinstance(c, dict) or not str(c.get("text") or "").strip():
                continue
            goto = c.get("goto") if c.get("goto") in known else f"gen_{seed_from(sid, j):08x}"
            choice: dict[str, Any] = {
                "id": f"{sid}_{j + 1}",
                "text": str(c["text"]).strip()[:200],
                "goto": goto,
            }
            sets = {}
            for eff in c.get("effects") or []:
                if isinstance(eff, dict) and eff.get("var") and isinstance(eff.get("add"), int):
                    name = slug(eff["var"])
                    sets[name] = {
                        "op": "+",
                        "args": [
                            {"op": "if", "args": [{"var": name}, {"var": name}, 0]},
                            eff["add"],
                        ],
                    }
            if sets:
                choice["set"] = sets
            scene["choices"].append(choice)
        if not scene["choices"]:
            scene["choices"] = [
                {"id": f"{sid}_1", "text": "Continue", "goto": f"gen_{seed_from(sid, 0):08x}"}
            ]
    return StoryScene.model_validate(scene).model_dump(by_alias = True, exclude_none = True)


# ---------------------------------------------------------------------------
# Entry point used by GET /api/flows/{id}/story
# ---------------------------------------------------------------------------
def compile_story_report(doc: dict, node_outputs: Optional[dict] = None) -> tuple[dict, list[str]]:
    node_outputs = node_outputs or {}
    warnings: list[str] = []
    picked = most_downstream(doc, node_outputs, STORY_SOURCE_TYPES, "story")
    manifest: Optional[dict] = None
    if picked is not None and isinstance(picked[1], dict):
        manifest = dict(picked[1])
    else:
        manifest = compile_authored(doc, node_outputs, warnings)
    if manifest is None:
        raise StoryCompileError(
            "This flow has no story yet: add Story and Scene nodes, or run a Story Generator."
        )
    eps = dict(manifest.get("entry_points") or {})
    eps.update({k: v for k, v in _entry_points(doc, STORY_ENTRY_TYPES).items() if k not in eps})
    manifest["entry_points"] = eps
    if manifest.get("mode") == "infinite" and "next_scene" not in eps:
        warnings.append(
            "Infinite mode needs a Next Scene (live) node; playing the written scenes only"
        )
        manifest["mode"] = "live"
    if "scene_art" not in eps and any(
        s.get("generate_background") for s in manifest.get("scenes") or []
    ):
        warnings.append(
            "Scenes ask for generated backgrounds but the flow has no Scene Art (live) node"
        )
    infinite = manifest.get("mode") == "infinite"
    return finalize(manifest, warnings, allow_unknown_gotos = infinite), warnings


def compile_story(doc: dict, node_outputs: Optional[dict] = None) -> dict:
    return compile_story_report(doc, node_outputs)[0]


def walk_check(manifest: dict) -> list[str]:
    """Problems a player would hit: unreachable scenes and scenes that can never reach an ending."""
    scenes = {s["id"]: s for s in manifest["scenes"]}
    seen, stack = set(), [manifest["start"]]
    while stack and len(seen) < 10_000:
        cur = stack.pop()
        if cur in seen or cur not in scenes:
            continue
        seen.add(cur)
        stack.extend(_targets(scenes[cur]))
    problems = [f"unreachable scene {sid}" for sid in scenes if sid not in seen]
    good = _reaches_ending(scenes)
    problems += [f"scene {sid} never reaches an ending" for sid in seen if sid not in good]
    return problems
