// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Sims-style needs: decay with the game clock, places advertise what they restore. Pure and seeded.

import type { WorldPlace } from "../../types";
import type { PlanBlock } from "./world-sim";

export const NEEDS = ["energy", "hunger", "social", "fun"] as const;
export type Need = (typeof NEEDS)[number];
export type Needs = Record<Need, number>;

export type ActionKind =
  | "go_to"
  | "eat"
  | "sleep"
  | "work"
  | "socialise"
  | "wander"
  | "idle";

/** Must match the backend's allowed conversation actions. */
export const REPLY_ACTIONS = [
  "follow_player",
  "stop_following",
  "go_to",
] as const;
export type ReplyAction = (typeof REPLY_ACTIONS)[number];

export interface Doing {
  kind: ActionKind;
  place: string;
  until: number;
  label?: string;
}

/** Points lost per game minute: hunger empties in about 5 hours, energy in about a day. */
const DECAY: Needs = { energy: 0.07, hunger: 0.3, social: 0.2, fun: 0.16 };

interface ActionDef {
  minutes: number;
  gain: Partial<Needs>;
  icon: string;
  says: string[];
}

export const ACTIONS: Record<ActionKind, ActionDef> = {
  go_to: { minutes: 60, gain: {}, icon: "", says: [] },
  eat: {
    minutes: 30,
    gain: { hunger: 2.5 },
    icon: "🍞",
    says: ["Time for a bite.", "I'm starving!", "Something smells good."],
  },
  sleep: {
    minutes: 360,
    gain: { energy: 0.35, hunger: 0.22 },
    icon: "💤",
    says: ["I can barely keep my eyes open.", "Off to bed."],
  },
  work: {
    minutes: 120,
    gain: { energy: -0.03, fun: -0.05 },
    icon: "🔨",
    says: ["Back to work.", "Plenty to do today."],
  },
  socialise: {
    minutes: 45,
    gain: { social: 1.6, fun: 0.3 },
    icon: "💬",
    says: ["I could use some company.", "Wonder who's about."],
  },
  wander: {
    minutes: 40,
    gain: { fun: 1.1 },
    icon: "🚶",
    says: ["Lovely day for a stroll.", "Let's see what's going on."],
  },
  idle: { minutes: 20, gain: {}, icon: "", says: [] },
};

const ADVERTS: [RegExp, Partial<Record<ActionKind, number>>][] = [
  [/home|house|cottage|hut|cabin|bed|tent|camp|inn|lodge|dorm/i, { sleep: 1 }],
  [
    /cafe|café|bak|tavern|inn|pub|bar|kitchen|diner|restaurant|food|market|orchard|farm|mess|canteen/i,
    { eat: 1 },
  ],
  [
    /square|plaza|tavern|pub|bar|market|hall|well|fountain|cafe|café|green|commons/i,
    { socialise: 1 },
  ],
  [
    /park|garden|pier|beach|lake|pond|meadow|forest|wood|grove|library|field|river|hill|shore|dock/i,
    { wander: 1 },
  ],
  [
    /shop|forge|smith|mill|farm|workshop|office|school|dock|stall|store|lab|clinic|studio|mine|harbo/i,
    { work: 1 },
  ],
];

export interface Offer {
  kind: ActionKind;
  place: string;
  strength: number;
}

/** An action no place names is offered everywhere at half strength. */
export function advertisements(places: WorldPlace[]): Offer[] {
  const out: Offer[] = [];
  const seen = new Set<ActionKind>();
  for (const p of places) {
    for (const [re, offers] of ADVERTS) {
      if (!re.test(p.name) && !re.test(p.id)) continue;
      for (const [kind, strength] of Object.entries(offers) as [
        ActionKind,
        number,
      ][]) {
        if (out.some((o) => o.kind === kind && o.place === p.id)) continue;
        out.push({ kind, place: p.id, strength });
        seen.add(kind);
      }
    }
  }
  for (const kind of ["eat", "sleep", "socialise", "wander"] as ActionKind[]) {
    if (seen.has(kind)) continue;
    for (const p of places) out.push({ kind, place: p.id, strength: 0.5 });
  }
  return out;
}

const clamp = (v: number) => Math.max(0, Math.min(100, v));

export function clampNeeds(raw: unknown, fallback: Needs): Needs {
  const r = (raw ?? {}) as Partial<Record<Need, unknown>>;
  const out = { ...fallback };
  for (const n of NEEDS) {
    const v = r[n];
    if (typeof v === "number" && Number.isFinite(v)) out[n] = clamp(v);
  }
  return out;
}

/** mulberry32 over a string hash: the same NPC at the same minute always draws the same number. */
export function seeded(...parts: (string | number)[]): () => number {
  let h = 2166136261;
  for (const ch of parts.join("|")) {
    h ^= ch.charCodeAt(0);
    h = Math.imul(h, 16777619);
  }
  let a = h >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function initialNeeds(seed: number, npcId: string): Needs {
  const rnd = seeded(seed, npcId, "needs");
  return {
    energy: 70 + rnd() * 25,
    hunger: 55 + rnd() * 35,
    social: 50 + rnd() * 40,
    fun: 50 + rnd() * 40,
  };
}

export function tickNeeds(
  needs: Needs,
  minutes: number,
  doing: ActionKind | null,
  atPlace: boolean,
): Needs {
  const gain = doing && atPlace ? ACTIONS[doing].gain : {};
  const out = { ...needs };
  for (const n of NEEDS)
    out[n] = clamp(needs[n] + ((gain[n] ?? 0) - DECAY[n]) * minutes);
  return out;
}

/** Low needs shout louder: a need at 20 weighs 0.64, one at 80 only 0.04. */
const urgency = (v: number) => ((100 - clamp(v)) / 100) ** 2;

const SERVES: Record<ActionKind, Need[]> = {
  go_to: [],
  eat: ["hunger"],
  sleep: ["energy"],
  work: [],
  socialise: ["social", "fun"],
  wander: ["fun"],
  idle: [],
};

export const PLAN_BONUS = 0.35;

function score(
  offer: Offer,
  needs: Needs,
  hour: number,
  planPlace: string | null,
): number {
  let s = 0;
  for (const n of SERVES[offer.kind]) s += urgency(needs[n]) * offer.strength;
  if (offer.kind === "sleep" && (hour >= 22 || hour < 6)) s *= 2;
  if (offer.kind === "work" && hour >= 9 && hour < 17) s += 0.12;
  if (planPlace && offer.place === planPlace) s += 0.1;
  return s;
}

/** Draws among the top three offers in proportion to score, so villagers are not clockwork. */
export function chooseAction(opts: {
  needs: Needs;
  offers: Offer[];
  clock: number;
  block: PlanBlock | null;
  rnd: () => number;
}): Doing {
  const { needs, offers, clock, block, rnd } = opts;
  const hour = Math.floor((clock % 1440) / 60);
  const scored = offers.map((o) => ({
    o,
    s: score(o, needs, hour, block?.place ?? null),
  }));
  if (block)
    scored.push({
      o: { kind: "go_to", place: block.place, strength: 1 },
      s: PLAN_BONUS,
    });
  scored.sort((a, b) => b.s - a.s);
  const best = scored[0]?.s ?? 0;
  const top = scored.slice(0, 3).filter((x) => x.s > 0 && x.s >= best * 0.6);
  if (!top.length) {
    const place = offers[0]?.place ?? block?.place ?? "";
    return { kind: "idle", place, until: clock + ACTIONS.idle.minutes };
  }
  const total = top.reduce((a, x) => a + x.s, 0);
  let r = rnd() * total;
  let pick = top[top.length - 1];
  for (const x of top) {
    r -= x.s;
    if (r <= 0) {
      pick = x;
      break;
    }
  }
  const kind = pick.o.kind;
  const minutes =
    kind === "go_to" && block
      ? Math.max(10, Math.min(90, block.start + block.minutes - clock))
      : ACTIONS[kind].minutes;
  return {
    kind,
    place: pick.o.place,
    until: clock + minutes,
    ...(kind === "go_to" && block ? { label: block.activity } : {}),
  };
}

export function satisfied(doing: Doing, needs: Needs): boolean {
  const served = SERVES[doing.kind];
  return served.length > 0 && served.every((n) => needs[n] >= 98);
}

export function parseDoing(raw: unknown): Doing | null {
  const r = raw as Partial<Doing> | null;
  if (
    !r ||
    typeof r !== "object" ||
    typeof r.kind !== "string" ||
    !(r.kind in ACTIONS) ||
    typeof r.place !== "string" ||
    typeof r.until !== "number" ||
    !Number.isFinite(r.until)
  )
    return null;
  return {
    kind: r.kind as ActionKind,
    place: r.place,
    until: r.until,
    ...(typeof r.label === "string" ? { label: r.label.slice(0, 120) } : {}),
  };
}

export function followArea(p: { x: number; y: number }) {
  return { x: Math.max(0, p.x - 1), y: Math.max(0, p.y - 1), w: 3, h: 3 };
}
