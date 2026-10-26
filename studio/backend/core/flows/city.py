# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mayor mode: the `city` block of a world and the one model call a month (advisor, petitions, an event, a name).

The simulation itself runs in the player (city-sim.ts). The model only writes words grounded in the numbers the
player sends; it never places tiles. The system prompt is a constant so every call shares its prefix."""

from __future__ import annotations

import re
from typing import Any, Optional

CITY_EVENTS = ("none", "festival", "fire", "strike", "flood", "boom")
CITY_METRICS = ("land_value", "pollution", "crime", "traffic", "power", "road")
ZONE_KINDS = {"R": "residential", "C": "commercial", "I": "industrial", "P": "power plant"}
_ZONE_ID = re.compile(r"^[RCIP][0-9]{1,4}$")
_ZONE_REF = re.compile(r"\b([RCIP][0-9]{1,4})\b")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_COORD = re.compile(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)|\b[xy]\s*=\s*\d")
MAX_ZONES = 24
MAX_RESIDENTS = 8

MAYOR_SYSTEM = """You advise the mayor of a small city in a city-building game and speak for its residents.
You only write words. The game places every road, power line and zone itself: never give coordinates and never ask
for a building at a particular tile. Use only numbers that appear in the report, copied exactly, and refer to zones
only by their ids from the report (like R1 or C2).
- advice: 2 or 3 short sentences on the most pressing problem and what kind of thing to build or change.
- petition_1_* and petition_2_*: when Petitions is on, up to two different residents from the Residents list each
  ask for one thing about their home. citizen is the resident id, metric is one of land_value, pollution, crime,
  traffic, power, road, and text is one sentence in their voice that quotes the exact number for that metric
  from their home line (power and road are 1 for yes, 0 for no). Leave citizen, metric and text empty for an unused
  petition and whenever Petitions is off.
- event: when Events is on you may rarely pick one event from this menu when the numbers suggest it: festival
  (lifts residential and commercial demand), boom (lifts industrial demand), fire or flood (damages one zone),
  strike (one commercial or industrial zone). Otherwise none. event_zone is the zone id for fire, flood or strike,
  else empty; event_size is 1 to 3, or 0 for none.
- name_zone and name: when Name zone gives an id, set name_zone to it and name to a short friendly district name.
  Otherwise leave both empty.
Answer in JSON."""

MAYOR_SCHEMA = {
    "type": "object",
    "properties": {
        "advice": {"type": "string"},
        "petition_1_citizen": {"type": "string"},
        "petition_1_metric": {"type": "string", "enum": ["", *CITY_METRICS]},
        "petition_1_text": {"type": "string"},
        "petition_2_citizen": {"type": "string"},
        "petition_2_metric": {"type": "string", "enum": ["", *CITY_METRICS]},
        "petition_2_text": {"type": "string"},
        "event": {"type": "string", "enum": list(CITY_EVENTS)},
        "event_zone": {"type": "string"},
        "event_size": {"type": "integer", "minimum": 0, "maximum": 3},
        "name_zone": {"type": "string"},
        "name": {"type": "string"},
    },
    "required": [
        "advice",
        "petition_1_citizen",
        "petition_1_metric",
        "petition_1_text",
        "petition_2_citizen",
        "petition_2_metric",
        "petition_2_text",
        "event",
        "event_zone",
        "event_size",
        "name_zone",
        "name",
    ],
    "additionalProperties": False,
}

_NUMBERS = (
    "month",
    "funds",
    "tax",
    "tax_income",
    "road_cost",
    "cash_flow",
    "residential_pop",
    "commercial_pop",
    "industrial_pop",
    "total_pop",
    "demand_r",
    "demand_c",
    "demand_i",
    "land_value_avg",
    "pollution_avg",
    "crime_avg",
    "power_capacity",
    "power_used",
    "trips_ok",
    "trips_failed",
    "trips_no_road",
)


class CityError(ValueError):
    pass


def normalize_city(raw: Any) -> Optional[dict]:
    """`city: true` or a dict turns Mayor mode on; anything else (or enabled false) leaves the world as it was."""
    if raw is True:
        raw = {}
    if not isinstance(raw, dict) or raw.get("enabled") is False:
        return None

    def num(key: str, lo: int, hi: int, default: int) -> int:
        v = raw.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return default
        return max(lo, min(hi, int(v)))

    return {
        "enabled": True,
        "funds": num("funds", 0, 1_000_000, 20000),
        "tax": num("tax", 0, 20, 7),
        **{k: raw.get(k) is not False for k in ("advisor", "petitions", "events")},
    }


def _int(
    v: Any,
    lo: int = -(10**9),
    hi: int = 10**9,
) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return 0
    return max(lo, min(hi, int(v)))


def _text(v: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()[:limit]


def clean_digest(state: dict) -> dict:
    raw = state.get("city")
    if not isinstance(raw, dict):
        raise CityError("The mayor's advisor needs the city report from the world player.")
    d: dict[str, Any] = {k: _int(raw.get(k)) for k in _NUMBERS}
    d["title"] = _text(raw.get("title"), 80) or "the city"
    d["kind"] = "advice" if raw.get("kind") == "advice" else "month"
    d["petitions"] = raw.get("petitions") is True and d["kind"] == "month"
    d["events"] = raw.get("events") is True and d["kind"] == "month"
    zones = []
    for z in raw.get("zones") or []:
        if not isinstance(z, dict) or not _ZONE_ID.match(str(z.get("id") or "")):
            continue
        if z.get("kind") not in ZONE_KINDS or any(x["id"] == z["id"] for x in zones):
            continue
        zones.append(
            {
                "id": z["id"],
                "kind": z["kind"],
                "pop": _int(z.get("pop"), 0, 40),
                "power": 1 if z.get("power") else 0,
                "road": 1 if z.get("road") else 0,
                "name": _text(z.get("name"), 40),
            }
        )
        if len(zones) >= MAX_ZONES:
            break
    d["zones"] = zones
    ids = {z["id"] for z in zones}
    residents = []
    for r in raw.get("residents") or []:
        if not isinstance(r, dict) or r.get("zone") not in ids:
            continue
        rid = _text(r.get("id"), 60)
        if not rid or any(x["id"] == rid for x in residents):
            continue
        residents.append(
            {
                "id": rid,
                "name": _text(r.get("name"), 60) or rid,
                "persona": _text(r.get("persona"), 120),
                "zone": r["zone"],
                **{m: _int(r.get(m), 0, 1000) for m in CITY_METRICS},
            }
        )
        if len(residents) >= MAX_RESIDENTS:
            break
    d["residents"] = residents
    unnamed = str(raw.get("unnamed") or "")
    d["unnamed"] = unnamed if d["kind"] == "month" and unnamed in ids and unnamed[0] != "P" else ""
    return d


def mayor_prompt(state: dict) -> tuple[str, str]:
    """The constant system prompt and a report: what rarely changes first, every number last."""
    d = clean_digest(state)
    people = "\n".join(
        f"- {r['id']}: {r['name']}" + (f", {r['persona']}" if r["persona"] else "")
        for r in d["residents"]
    )
    request = (
        "monthly report" if d["kind"] == "month" else "the mayor asks for advice now (advice only)"
    )
    zones = "\n".join(
        f"- {z['id']}{' ' + z['name'] if z['name'] else ''} ({ZONE_KINDS[z['kind']]}, pop {z['pop']}, "
        f"power {'yes' if z['power'] else 'no'}, road {'yes' if z['road'] else 'no'})"
        for z in d["zones"]
    )
    homes = "\n".join(
        f"- {r['id']} lives in {r['zone']}: " + ", ".join(f"{m} {r[m]}" for m in CITY_METRICS)
        for r in d["residents"]
    )
    prompt = (
        f"City: {d['title']}\n"
        f"Residents:\n{people or '- none yet'}\n"
        f"Request: {request}\n"
        f"Petitions: {'on' if d['petitions'] and d['residents'] else 'off'}\n"
        f"Events: {'on' if d['events'] else 'off'}\n"
        f"Name zone: {d['unnamed'] or 'none'}\n"
        f"Month {d['month']}\n"
        f"Funds {d['funds']}, tax rate {d['tax']}\n"
        f"Last month: tax income {d['tax_income']}, road upkeep {d['road_cost']}, cash flow {d['cash_flow']}\n"
        f"Population: residential {d['residential_pop']}, commercial {d['commercial_pop']}, industrial "
        f"{d['industrial_pop']}, total {d['total_pop']}\n"
        f"Demand: residential {d['demand_r']}, commercial {d['demand_c']}, industrial {d['demand_i']} "
        "(-2000 to 2000)\n"
        f"Averages: land value {d['land_value_avg']}, pollution {d['pollution_avg']}, crime {d['crime_avg']}\n"
        f"Power: capacity {d['power_capacity']}, used {d['power_used']}\n"
        f"Trips last month: passed {d['trips_ok']}, failed {d['trips_failed']}, no road {d['trips_no_road']}\n"
        f"Zones:\n{zones or '- none'}\n"
        f"Resident homes:\n{homes or '- none'}\n"
    )
    return MAYOR_SYSTEM, prompt


def _numbers(text: str) -> set[int]:
    out = set()
    for m in _NUMBER.findall(text):
        try:
            out.add(int(float(m.replace(",", ""))))
        except ValueError:
            pass
    return out


def _digest_numbers(d: dict) -> set[int]:
    nums = {abs(d[k]) for k in _NUMBERS}
    for z in d["zones"]:
        nums |= {z["pop"], int(z["id"][1:])}
    for r in d["residents"]:
        nums |= {r[m] for m in CITY_METRICS} | {int(r["zone"][1:])}
    # Counts the advice may state: zones of a kind, unpowered, without a road, residents.
    real = [z for z in d["zones"] if z["kind"] != "P"]
    nums |= {len(d["zones"]), len(d["residents"]), len(real)}
    nums |= {sum(1 for z in d["zones"] if z["kind"] == k) for k in ZONE_KINDS}
    nums |= {sum(1 for z in real if not z["power"]), sum(1 for z in real if not z["road"])}
    return nums


def local_advice(d: dict) -> str:
    zones = d["zones"]
    tips = []
    if not any(z["kind"] == "P" for z in zones):
        tips.append("Build a power plant: nothing grows without power.")
    dark = [z["id"] for z in zones if z["kind"] != "P" and not z["power"]]
    if dark and len(tips) == 0:
        tips.append(f"{len(dark)} zones have no power ({', '.join(dark[:3])}).")
    lost = [z["id"] for z in zones if z["kind"] != "P" and not z["road"]]
    if lost:
        tips.append(f"{len(lost)} zones have no road ({', '.join(lost[:3])}).")
    tips.append(
        f"Population {d['total_pop']}, funds {d['funds']}, cash flow {d['cash_flow']} a month."
    )
    return " ".join(tips)


def _grounded(text: str, d: dict) -> bool:
    if _COORD.search(text):
        return False
    ids = {z["id"] for z in d["zones"]}
    if any(ref not in ids for ref in _ZONE_REF.findall(text)):
        return False
    return _numbers(text) <= _digest_numbers(d)


def validate_mayor(raw: Any, state: dict) -> dict:
    """Keep what the numbers support; anything ungrounded falls back to fixed text, so the call is never retried."""
    d = clean_digest(state)
    raw = raw if isinstance(raw, dict) else {}
    advice = _text(raw.get("advice"), 600) if isinstance(raw.get("advice"), str) else ""
    source = "model"
    if not re.search(r"[A-Za-z]{3}", advice) or not _grounded(advice, d):
        advice, source = local_advice(d), "fallback"
    petitions = []
    if d["petitions"]:
        by_id = {r["id"]: r for r in d["residents"]}
        by_name = {r["name"].lower(): r for r in d["residents"]}
        for i in (1, 2):
            key = _text(raw.get(f"petition_{i}_citizen"), 60)
            who = by_id.get(key) or by_name.get(key.lower())
            metric = raw.get(f"petition_{i}_metric")
            if who is None or metric not in CITY_METRICS:
                continue
            if any(p["citizen"] == who["id"] for p in petitions):
                continue
            value = who[metric]
            label = metric.replace("_", " ")
            text = _text(raw.get(f"petition_{i}_text"), 240)
            if not text or _COORD.search(text) or not _numbers(text) <= {value}:
                text = f"Please do something about {label} at home in {who['zone']}: it is {value}."
            elif value not in _numbers(text):
                text = f"{text} ({label} {value})"
            petitions.append(
                {
                    "citizen": who["id"],
                    "zone": who["zone"],
                    "metric": metric,
                    "value": value,
                    "text": text,
                }
            )
    event = None
    kind = raw.get("event")
    if d["events"] and kind in CITY_EVENTS and kind != "none":
        zones = {z["id"]: z for z in d["zones"]}
        zone = zones.get(_text(raw.get("event_zone"), 8))
        size = max(1, min(3, _int(raw.get("event_size"))))
        if kind in ("festival", "boom"):
            event = {"kind": kind, "zone": "", "size": size}
        elif zone and (
            (kind in ("fire", "flood") and zone["kind"] != "P")
            or (kind == "strike" and zone["kind"] in ("C", "I"))
        ):
            event = {"kind": kind, "zone": zone["id"], "size": size}
    name = None
    given = re.sub(r"[^A-Za-z' -]", "", _text(raw.get("name"), 40)).strip()
    if d["unnamed"] and given and _text(raw.get("name_zone"), 8) in ("", d["unnamed"]):
        name = {"zone": d["unnamed"], "name": given}
    return {
        "advice": advice,
        "advice_source": source,
        "petitions": petitions,
        "event": event,
        "name": name,
    }
