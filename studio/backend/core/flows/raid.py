# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Raids authored once by the LLM: a camp theme, an enemy roster from a fixed menu, loot and a pool of short barks.

Returned as data for the hostile AI to read when present; nothing here touches gameplay."""

from __future__ import annotations

import re
from typing import Any, Optional

from core.flows.authoring import (
    AuthoringError,
    Authored,
    constrained_call,
    strict_schema,
    string_enum,
)
from core.flows.expr import Prng

ENEMY_KINDS = ("grunt", "scout", "brute", "archer", "captain")
BARK_EVENTS = ("spot", "lost", "hurt", "flank", "retreat")
LOOT = ("coins", "potion", "arrows", "bread", "lantern oil", "old key", "gem")
CAMP_THEMES = (
    "bandit camp",
    "goblin warren",
    "smugglers' cove",
    "ruined watchtower",
    "forest hideout",
)
MAX_ROSTER = 6
MAX_BARKS = 24
MIN_BARKS = 12
BARK_CHARS = 60

_BARKS = {
    "spot": ("There you are!", "Over here, lads!", "Intruder!", "Got you now!"),
    "lost": (
        "Where'd they go?",
        "Must have been the wind.",
        "Keep looking.",
        "They can't hide forever.",
    ),
    "hurt": ("Argh!", "That stings!", "Lucky shot!", "I'm hit!"),
    "flank": ("Go round the side!", "Cut them off!", "Circle left!", "Pin them down!"),
    "retreat": ("Fall back!", "Not worth it!", "Back to camp!", "Run for it!"),
}


def raid_schema(items: list[str]) -> dict:
    return strict_schema(
        {
            "type": "object",
            "properties": {
                "theme": string_enum(CAMP_THEMES),
                "camp_name": {"type": "string"},
                "roster": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_ROSTER,
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": string_enum(ENEMY_KINDS),
                            "count": {"type": "integer", "minimum": 1, "maximum": 4},
                        },
                        "required": ["kind", "count"],
                    },
                },
                "loot": {"type": "array", "maxItems": 4, "items": string_enum([*LOOT, *items])},
                "barks": {
                    "type": "array",
                    "minItems": MIN_BARKS,
                    "maxItems": MAX_BARKS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "event": string_enum(BARK_EVENTS),
                            "text": {"type": "string"},
                        },
                        "required": ["event", "text"],
                    },
                },
            },
            "required": ["theme", "camp_name", "roster", "loot", "barks"],
        }
    )


def validate_raid(raw: Any, items: Optional[list[str]] = None) -> dict:
    if not isinstance(raw, dict):
        raise AuthoringError("the answer is not a raid object")
    if raw.get("theme") not in CAMP_THEMES:
        raise AuthoringError(f"theme must be one of {', '.join(CAMP_THEMES)}")
    roster: dict[str, int] = {}
    for r in raw.get("roster") or []:
        if not isinstance(r, dict) or r.get("kind") not in ENEMY_KINDS:
            raise AuthoringError(f"roster kinds must be one of {', '.join(ENEMY_KINDS)}")
        roster[r["kind"]] = min(4, roster.get(r["kind"], 0) + max(1, int(r.get("count") or 1)))
    if not roster:
        raise AuthoringError("the roster is empty")
    allowed = {*LOOT, *(items or [])}
    loot = list(dict.fromkeys(x for x in raw.get("loot") or [] if x in allowed))[:4]
    barks: list[dict] = []
    seen: set[str] = set()
    for b in raw.get("barks") or []:
        if not isinstance(b, dict) or b.get("event") not in BARK_EVENTS:
            continue
        text = re.sub(r"\s+", " ", str(b.get("text") or "")).strip()[:BARK_CHARS]
        if text and text.lower() not in seen:
            seen.add(text.lower())
            barks.append({"event": b["event"], "text": text})
    barks = barks[:MAX_BARKS]
    missing = [e for e in BARK_EVENTS if not any(b["event"] == e for b in barks)]
    if len(barks) < MIN_BARKS or missing:
        raise AuthoringError(
            f"write at least {MIN_BARKS} different short barks covering every event"
            + (f" (none for {', '.join(missing)})" if missing else "")
        )
    return {
        "theme": raw["theme"],
        "camp_name": re.sub(r"\s+", " ", str(raw.get("camp_name") or "")).strip()[:60]
        or raw["theme"].title(),
        "roster": [{"kind": k, "count": n} for k, n in roster.items()],
        "loot": loot,
        "barks": barks,
    }


def fallback_raid(seed: int, items: Optional[list[str]] = None) -> dict:
    """The same raid for the same seed: a theme, a balanced roster, loot and every built-in bark."""
    rng = Prng(seed)
    theme = CAMP_THEMES[int(rng.random() * len(CAMP_THEMES))]
    size = 2 + int(rng.random() * 3)
    roster = [{"kind": "grunt", "count": size}, {"kind": "scout", "count": 1}]
    if size >= 3:
        roster.append({"kind": "captain", "count": 1})
    pool = [*LOOT, *(items or [])]
    loot = list(dict.fromkeys(pool[int(rng.random() * len(pool))] for _ in range(2)))
    barks = [{"event": e, "text": t} for e, lines in _BARKS.items() for t in lines]
    return validate_raid(
        {
            "theme": theme,
            "camp_name": theme.title(),
            "roster": roster,
            "loot": loot,
            "barks": barks,
        },
        items,
    )


RAID_SYSTEM = (
    "You plan a raid on a camp for a small action game. Answer in JSON: `theme` and a short `camp_name`; `roster`, "
    "the enemies as kind ("
    + ", ".join(ENEMY_KINDS)
    + ") and count; up to 4 `loot` names from the allowed list; "
    f"and {MIN_BARKS} to {MAX_BARKS} `barks`, short lines (under {BARK_CHARS} characters) enemies shout, each tagged "
    "with its event: spot (sees the player), lost (lost track), hurt, flank, retreat. Cover every event. No gore."
)


async def generate_raid(
    ctx: Any,
    world: Optional[dict] = None,
    hint: str = "",
    *,
    seed: int = 0,
    model: Optional[dict] = None,
) -> Authored:
    world = world or {}
    items = [
        str(i.get("name"))
        for i in world.get("items") or []
        if isinstance(i, dict) and i.get("name")
    ][:20]
    prompt = (
        f"World: {world.get('title') or 'a small village'}. {str(world.get('intro_text') or '')[:300]}\n"
        f"Allowed loot: {', '.join([*LOOT, *items])}\n"
        + (f"The raid should feel like: {hint[:300]}\n" if hint else "")
    )
    return await constrained_call(
        ctx,
        raid_schema(items),
        prompt,
        lambda raw: validate_raid(raw, items),
        fallback = lambda: fallback_raid(seed, items),
        system = RAID_SYSTEM,
        model = model,
        temperature = 0.9,
        max_tokens = 1200,
    )
