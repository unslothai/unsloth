# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Village Rivals: normalise the `rivals` block of a world and answer the two LLM calls of the mode.

The game itself runs in the player (`player/world/rivals.ts`). The strategist is one call per AI faction per
turn that returns posture weights, a target and goals; diplomacy is a leader's reply to the player, conditioned
on that intent, whose offer is checked against the engine's numbers so the text never promises what the
engine will not do."""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Optional

from core.flows.registry import FlowNodeError, NodeContext
from core.flows.story import _unique, slug

RESOURCES = ("food", "wood", "stone", "gold")
POSTURE_KEYS = ("economy", "expansion", "military", "diplomacy")
OFFER_KINDS = ("none", "trade", "ally", "peace")
MAX_FACTIONS = 4
DIGEST_CHARS = 2400
MAX_TRADE = 10
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
# A reply with no offer must not read as a promise the engine would have to keep.
_PROMISE = re.compile(
    r"\b(i|we)\s*(will|'ll|shall|promise to|agree to)\s+(give|send|trade|pay|ally|join|help|protect|stop|"
    r"make peace|fight)\b|\b(it'?s a deal|deal!|agreed|you have my word|we accept|i accept)\b",
    re.I,
)

STRATEGIST_SCHEMA = {
    "type": "object",
    "properties": {
        **{k: {"type": "number", "minimum": 0, "maximum": 1} for k in POSTURE_KEYS},
        "target": {"type": "string"},
        "goals": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "say": {"type": "string"},
    },
    "required": [*POSTURE_KEYS, "target", "goals", "say"],
    "additionalProperties": False,
}

DIPLOMACY_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "offer": {"type": "string", "enum": list(OFFER_KINDS)},
        "give": {"type": "string", "enum": ["", *RESOURCES]},
        "give_amount": {"type": "integer", "minimum": 0, "maximum": MAX_TRADE},
        "want": {"type": "string", "enum": ["", *RESOURCES]},
        "want_amount": {"type": "integer", "minimum": 0, "maximum": MAX_TRADE},
    },
    "required": ["text", "offer", "give", "give_amount", "want", "want_amount"],
    "additionalProperties": False,
}


def normalize_rivals(
    raw: Any,
    npc_ids: set[str],
    regions: list[dict],
    ref: Callable[[Any, set[str]], Optional[str]],
    taken: set[str],
) -> Optional[dict]:
    """2-4 factions with unique ids, at most one player, leaders and home regions that resolve; else None."""
    if not isinstance(raw, dict) or not isinstance(raw.get("factions"), list):
        return None
    by_region = {}
    for r in regions:
        by_region[r["id"]] = r["id"]
        by_region[str(r["name"]).lower()] = r["id"]
    seen: set[str] = set(taken)
    used_regions: set[str] = set()
    factions = []
    player = False
    for f in raw["factions"]:
        if not isinstance(f, dict) or not str(f.get("name") or "").strip():
            continue
        if len(factions) == MAX_FACTIONS:
            break
        name = str(f["name"]).strip()[:60]
        region = by_region.get(str(f.get("region") or "").strip()) or by_region.get(
            str(f.get("region") or "").strip().lower()
        )
        if region in used_regions:
            region = None
        is_player = f.get("player") is True and not player
        player = player or is_player
        out = {
            "id": _unique(slug(f.get("id") or name, "faction"), seen),
            "name": name,
            "player": is_player,
            "leader": ref(f.get("leader"), npc_ids),
            "region": region,
            "persona": str(f.get("persona") or "")[:400],
            "color": f["color"]
            if isinstance(f.get("color"), str) and _HEX.match(f["color"])
            else None,
        }
        if region:
            used_regions.add(region)
        factions.append(out)
    if len(factions) < 2:
        return None
    free = [r["id"] for r in regions if r["id"] not in used_regions]
    for f in factions:
        if f["region"] is None and free:
            f["region"] = free.pop(0)
    turns = raw.get("max_turns")
    hyst = raw.get("hysteresis")
    return {
        "factions": factions,
        "max_turns": max(5, min(200, int(turns)))
        if isinstance(turns, (int, float)) and not isinstance(turns, bool)
        else 30,
        "hysteresis": max(0.0, min(1.0, float(hyst)))
        if isinstance(hyst, (int, float)) and not isinstance(hyst, bool)
        else 0.15,
    }


def _clip(text: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def _context(state: Any) -> tuple[dict, dict, dict]:
    if not isinstance(state, dict):
        raise FlowNodeError("Village Rivals needs the game state from the world player.")
    world = state.get("world") if isinstance(state.get("world"), dict) else {}
    factions = (world.get("rivals") or {}).get("factions") or []
    me = next(
        (f for f in factions if isinstance(f, dict) and f.get("id") == state.get("faction")), None
    )
    if me is None:
        raise FlowNodeError("Village Rivals: unknown faction.")
    digest = state.get("digest") if isinstance(state.get("digest"), dict) else {}
    return world, me, digest


def digest_text(digest: dict) -> str:
    """The digest as compact JSON, cut to a fixed size so a long game never grows the prompt."""
    text = json.dumps(digest, separators = (",", ":"), ensure_ascii = False, default = str)
    return text[:DIGEST_CHARS]


def _faction_ids(world: dict) -> dict[str, str]:
    out = {}
    for f in (world.get("rivals") or {}).get("factions") or []:
        if isinstance(f, dict) and f.get("id"):
            out[str(f["id"]).lower()] = f["id"]
            out[str(f.get("name") or "").lower()] = f["id"]
    return out


def _leader_line(state: dict, me: dict) -> str:
    leader = state.get("leader") if isinstance(state.get("leader"), dict) else None
    who = f"{_clip(leader.get('name'), 60)}, {_clip(leader.get('persona'), 300)}" if leader else ""
    persona = _clip(me.get("persona"), 400)
    return " ".join(x for x in (f"Its leader is {who}." if who else "", persona) if x)


def strategist_prompt(state: Any) -> tuple[str, str]:
    world, me, digest = _context(state)
    system = (
        f"You are the strategist of {_clip(me.get('name'), 60)}, a village in the game "
        f'"{_clip(world.get("title"), 120)}". {_leader_line(state, me)}\n'
        "Each day you set the village's posture: four weights from 0 to 1 for economy, expansion, military "
        "and diplomacy (they are normalised), a `target` village id to focus on (or empty), 1-3 short goals, "
        "and `say`, one short public line from the leader. Workers, building, raids and trades are carried out "
        "for you from these weights. Keep a plan for several days unless the situation changes. Answer in JSON."
    )
    prompt = f"State of the game (JSON):\n{digest_text(digest)}\n"
    return system, prompt


def validate_strategist(raw: Any, state: Any) -> dict:
    """Repair any answer into a plan: weights in [0, 1] normalised, target a rival id or None, 1-3 goals."""
    world, me, _ = _context(state)
    raw = raw if isinstance(raw, dict) else {}
    weights = {}
    for k in POSTURE_KEYS:
        v = raw.get(k)
        weights[k] = (
            max(0.0, min(1.0, float(v)))
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            else 0.0
        )
    total = sum(weights.values())
    posture = (
        {k: round(v / total, 3) for k, v in weights.items()}
        if total > 0
        else {k: 0.25 for k in POSTURE_KEYS}
    )
    target = _faction_ids(world).get(_clip(raw.get("target"), 60).lower())
    if target == me["id"]:
        target = None
    goals = (
        [g for g in (_clip(x, 60) for x in (raw.get("goals") or [])[:3]) if g]
        if isinstance(raw.get("goals"), list)
        else []
    )
    return {
        "posture": posture,
        "target": target,
        "goals": goals or ["grow the village"],
        "say": _clip(raw.get("say"), 120),
    }


def _intent_text(state: dict, world: dict) -> str:
    intent = state.get("intent") if isinstance(state.get("intent"), dict) else {}
    posture = intent.get("posture") if isinstance(intent.get("posture"), dict) else {}
    top = max(POSTURE_KEYS, key = lambda k: float(posture.get(k) or 0)) if posture else "economy"
    names = {f["id"]: f.get("name") for f in (world.get("rivals") or {}).get("factions") or []}
    target = intent.get("target")
    goals = "; ".join(_clip(g, 60) for g in (intent.get("goals") or [])[:3])
    return (
        f"Your village leans to {top}"
        + (f" and is set against {names.get(target, target)}" if target else "")
        + (f". Goals: {goals}" if goals else "")
        + "."
    )


def diplomacy_prompt(state: Any) -> tuple[str, str]:
    """The system prompt is fixed per village; intent, talk and the digest (last) go in the prompt, so a
    llama-server prompt cache keeps the shared prefix across messages and turns."""
    world, me, digest = _context(state)
    player = state.get("player")
    against_player = (state.get("intent") or {}).get("target") == player
    system = (
        f'You are the leader of {_clip(me.get("name"), 60)} in the game "{_clip(world.get("title"), 120)}", '
        f"talking with the player's village. {_leader_line(state, me)}\n"
        "Stay in character and keep to your village's intent. Reply in under 60 words as `text`. Put any deal "
        "in `offer`: trade (you give `give_amount` of `give` and want `want_amount` of `want`, at most "
        f"{MAX_TRADE} each, only goods you have), ally, peace, or none; unused fields are \"\" and 0. Never "
        "promise anything in `text` that is not in `offer`. Answer in JSON."
    )
    history = [
        f"{'Player' if h.get('role') == 'player' else 'You'}: {_clip(h.get('text'), 300)}"
        for h in (state.get("history") or [])[-6:]
        if isinstance(h, dict)
    ]
    prompt = (
        _intent_text(state, world)
        + (" You are set against the player: do not offer an alliance." if against_player else "")
        + "\n"
        + ("Conversation so far:\n" + "\n".join(history) + "\n" if history else "")
        + f"Player: {_clip(state.get('player_message'), 500)}\n"
        + f"State of the game (JSON):\n{digest_text(digest)}\n"
    )
    return system, prompt


def _decline(state: dict, world: dict) -> str:
    intent = state.get("intent") if isinstance(state.get("intent"), dict) else {}
    if intent.get("target") and intent.get("target") == state.get("player"):
        return "We have nothing to agree on with you right now."
    return "Not today. Our village has other plans."


def validate_diplomacy(raw: Any, state: Any) -> dict:
    """Keep the offer only when the leader's village can honour it and it fits the intent; drop promises
    the text makes that the offer does not back."""
    world, me, digest = _context(state)
    raw = raw if isinstance(raw, dict) else {}
    text = _clip(raw.get("text"), 400)
    kind = raw.get("offer") if raw.get("offer") in OFFER_KINDS else "none"
    stock = ((digest.get("you") or {}).get("stock")) or {}
    intent = state.get("intent") if isinstance(state.get("intent"), dict) else {}
    player = state.get("player")
    relation = next(
        (
            o.get("relation")
            for o in digest.get("others") or []
            if isinstance(o, dict) and o.get("id") == player
        ),
        "peace",
    )
    offer: Optional[dict] = None
    refused = ""
    if kind == "trade":
        give, want = raw.get("give"), raw.get("want")
        ga, wa = raw.get("give_amount"), raw.get("want_amount")
        ok_amount = all(
            isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= MAX_TRADE for n in (ga, wa)
        )
        if give not in RESOURCES or want not in RESOURCES or give == want or not ok_amount:
            refused = "the trade is not well formed"
        elif not isinstance(stock.get(give), (int, float)) or stock[give] < ga:
            refused = f"the village does not have {ga} {give}"
        elif relation == "war":
            refused = "the villages are at war"
        else:
            offer = {
                "kind": "trade",
                "give": give,
                "give_amount": ga,
                "want": want,
                "want_amount": wa,
            }
    elif kind == "ally":
        if intent.get("target") == player:
            refused = "the village is set against the player"
        elif relation != "peace":
            refused = "an alliance needs peace first"
        else:
            offer = {"kind": "ally"}
    elif kind == "peace":
        if relation != "war":
            refused = "the villages are not at war"
        else:
            offer = {"kind": "peace"}
    filtered = False
    if offer is None and (refused or _PROMISE.search(text) or not text):
        text = _decline(state, world)
        filtered = True
    return {
        "text": text,
        "offer": offer or {"kind": "none"},
        **({"filtered": True} if filtered else {}),
        **({"refused": refused} if refused else {}),
    }


async def strategist(ctx: NodeContext, state: dict) -> dict:
    """Exactly one model call: a bad answer is repaired, never asked again, so a turn costs one call per faction."""
    system, prompt = strategist_prompt(state)
    raw = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = system,
        prompt = prompt,
        schema = STRATEGIST_SCHEMA,
        temperature = 0.7,
        max_tokens = 300,
    )
    return {"reply": {"plan": validate_strategist(raw, state)}}


async def diplomacy(ctx: NodeContext, state: dict) -> dict:
    system, prompt = diplomacy_prompt(state)
    raw = await ctx.llm_json(
        model = ctx.params.get("model"),
        system = system,
        prompt = prompt,
        schema = DIPLOMACY_SCHEMA,
        temperature = 0.8,
        max_tokens = 300,
    )
    return {"reply": validate_diplomacy(raw, state)}


RIVAL_TASKS = {"rivals_strategist": strategist, "rivals_diplomacy": diplomacy}
