// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Opt-in villager cost cutters after Lyfe Agents and Affordable Generative Agents.

import type { WorldSpec } from "../../types";
import {
  ACTIONS,
  type ActionKind,
  type Doing,
  NEEDS,
  type Need,
  type Needs,
  seeded,
} from "./needs";
import type { Area } from "./phaser-game";
import {
  type MemoryRecord,
  NOTABLE,
  type Note,
  type NpcMind,
  type PairMemory,
  type PlanBlock,
  type PlanOption,
  type SimJob,
  type Until,
  clockLabel,
  currentBlock,
  dayOf,
  emptyMind,
  needsOn,
  remember,
  simOf,
} from "./world-sim";
import type { WorldState } from "./world-state";

export interface LifeConfig {
  options: boolean;
  cache: number;
  reflectAt: number;
  rateBatch: number;
  relationships: boolean;
}

export interface CacheEntry {
  key: string;
  /** Times are minutes into the day. */
  plan: PlanBlock[];
  options?: PlanOption[];
}

export interface LifeState {
  cache: CacheEntry[];
}

export const DAY_END = 22 * 60;
export const REPLAN_GAP = 60;
/** Options running out later than this leave the evening to needs alone, with no model call. */
export const LATE_REPLAN = DAY_END - 240;
/** A need, item or place option that has not ended after this long gives up. */
export const OPTION_MAX_MIN = 240;
export const CRITICAL = 15;
export const RATE_BATCH = 8;
export const LOW_NEED = 30;
const MAX_FACTS = 3;
const MAX_CHAIN = 4;

export function lifeOf(spec: WorldSpec): LifeConfig {
  const sim = simOf(spec);
  const pos = (v: unknown) =>
    typeof v === "number" && Number.isFinite(v) && v > 0 ? Math.floor(v) : 0;
  return {
    options: !!sim && sim.options === true && needsOn(spec),
    cache: sim ? Math.min(256, pos(sim.plan_cache)) : 0,
    reflectAt: sim ? pos(sim.reflect_at) : 0,
    rateBatch: Math.max(1, Math.min(32, pos(sim?.rate_batch) || RATE_BATCH)),
    relationships: !!sim && sim.relationships === true,
  };
}

export function freshLife(cfg: LifeConfig): Partial<NpcMind> {
  return {
    ...(cfg.reflectAt ? { since: 0 } : {}),
    ...(cfg.relationships ? { rel: {} } : {}),
  };
}

const inside = (a: Area | undefined, t: { x: number; y: number } | null) =>
  !!a && !!t && t.x >= a.x && t.x < a.x + a.w && t.y >= a.y && t.y < a.y + a.h;

export interface OptionCtx {
  npcId: string;
  clock: number;
  needs?: Needs;
  held?: Record<string, string>;
  at: { x: number; y: number } | null;
  areas: Map<string, Area>;
}

export function untilMet(u: Until, c: OptionCtx): boolean {
  switch (u.type) {
    case "time":
      return c.clock >= u.at;
    case "need":
      return !!c.needs && c.needs[u.need] >= u.above;
    case "item":
      return c.held?.[u.item] === c.npcId;
    case "place":
      return inside(c.areas.get(u.place), c.at);
  }
}

export function activeOption(mind: NpcMind): PlanOption | null {
  return mind.options?.[mind.optionAt ?? 0] ?? null;
}

export function advanceOptions(
  mind: NpcMind,
  c: OptionCtx,
): Pick<NpcMind, "optionAt" | "optionSince"> | null {
  if (!mind.options) return null;
  let at = mind.optionAt ?? 0;
  let since = mind.optionSince ?? c.clock;
  while (at < mind.options.length) {
    const o = mind.options[at];
    const stale = o.until.type !== "time" && c.clock - since >= OPTION_MAX_MIN;
    if (!untilMet(o.until, c) && !stale) break;
    at++;
    since = c.clock;
  }
  return at === (mind.optionAt ?? 0)
    ? null
    : { optionAt: at, optionSince: since };
}

export function optionDoing(o: PlanOption, clock: number): Doing {
  return {
    kind: o.kind,
    place: o.place,
    until: clock + Math.min(60, ACTIONS[o.kind].minutes),
    label: o.activity,
  };
}

export const critical = (needs: Needs) =>
  NEEDS.some((n) => needs[n] < CRITICAL);

export function lifeBlock(mind: NpcMind, clock: number): PlanBlock | null {
  const o = activeOption(mind);
  if (o) {
    const start = mind.optionSince ?? clock;
    const end = o.until.type === "time" ? o.until.at : start + OPTION_MAX_MIN;
    return {
      start,
      minutes: Math.max(10, end - start),
      activity: o.activity,
      place: o.place,
    };
  }
  return mind.planDay === dayOf(clock) ? currentBlock(mind.plan, clock) : null;
}

export function planDue(
  cfg: LifeConfig,
  mind: NpcMind,
  clock: number,
): boolean {
  if (mind.planDay !== dayOf(clock)) return true;
  if (!cfg.options || !mind.options) return false;
  if (clock - (mind.planAt ?? Number.NEGATIVE_INFINITY) < REPLAN_GAP)
    return false;
  if (mind.replan) return true;
  return (
    (mind.optionAt ?? 0) >= mind.options.length && clock % 1440 < LATE_REPLAN
  );
}

export function planAsk(cfg: LifeConfig, mind: NpcMind | undefined) {
  return cfg.options ? { options: true, needs: mind?.needs } : {};
}

function parseUntil(raw: unknown): Until | null {
  const u = raw as Record<string, unknown> | null;
  if (!u || typeof u !== "object") return null;
  if (u.type === "time" && typeof u.at === "number" && Number.isFinite(u.at))
    return { type: "time", at: u.at };
  if (
    u.type === "need" &&
    NEEDS.includes(u.need as Need) &&
    typeof u.above === "number"
  )
    return { type: "need", need: u.need as Need, above: u.above };
  if (u.type === "item" && typeof u.item === "string")
    return { type: "item", item: u.item };
  if (u.type === "place" && typeof u.place === "string")
    return { type: "place", place: u.place };
  return null;
}

export function parseOptions(raw: unknown): PlanOption[] | null {
  if (!Array.isArray(raw)) return null;
  const out: PlanOption[] = [];
  for (const o of raw as Record<string, unknown>[]) {
    const until = parseUntil(o?.until);
    if (
      !until ||
      typeof o.place !== "string" ||
      typeof o.activity !== "string" ||
      !(typeof o.kind === "string" && o.kind in ACTIONS)
    )
      continue;
    out.push({
      activity: o.activity.slice(0, 80),
      place: o.place,
      kind: o.kind as ActionKind,
      until,
    });
  }
  return out;
}

/** Mind fields after a plan answer (`out` null = failed call); a replan raised after `askedSeq` stays pending. */
export function lifePlanned(
  mind: NpcMind,
  out: unknown,
  clock: number,
  askedSeq: number = mind.replanSeq ?? 0,
): Partial<NpcMind> {
  const options = parseOptions((out as { options?: unknown } | null)?.options);
  const seen = (mind.replanSeq ?? 0) === askedSeq;
  if (options)
    return {
      options,
      optionAt: 0,
      optionSince: clock,
      planAt: clock,
      ...(seen ? { replan: false } : {}),
    };
  return mind.options ? { planAt: clock } : {};
}

const OCCUPATIONS =
  /\b(baker|farmer|fisher(?:man|woman)?|smith|blacksmith|merchant|trader|shopkeeper|guard|teacher|healer|doctor|priest|innkeeper|barkeep|bartender|carpenter|miner|hunter|tailor|librarian|scholar|artist|painter|musician|cook|chef|gardener|mayor|student|child|elder|sailor|scientist|mechanic|nurse)\b/i;

export function roleOf(spec: WorldSpec, npcId: string): string {
  const n = spec.npcs.find((x) => x.id === npcId);
  if (!n) return "villager";
  if (n.role) return n.role;
  const m = OCCUPATIONS.exec(`${n.persona ?? ""} ${n.routine ?? ""}`);
  return m ? m[1].toLowerCase() : "villager";
}

export function needsBucket(needs: Needs | undefined): string {
  if (!needs) return "-";
  let low: Need = NEEDS[0];
  for (const n of NEEDS) if (needs[n] < needs[low]) low = n;
  return needs[low] < LOW_NEED ? `${low}-low` : "ok";
}

export function timeBucket(clock: number): string {
  const h = Math.floor((clock % 1440) / 60);
  if (h < 6 || h >= 22) return "night";
  return h < 12 ? "morning" : h < 17 ? "afternoon" : "evening";
}

export function planKey(
  spec: WorldSpec,
  npcId: string,
  mind: NpcMind | undefined,
  clock: number,
  place: string,
): string {
  return [
    roleOf(spec, npcId),
    needsBucket(mind?.needs),
    timeBucket(clock),
    place,
  ].join("|");
}

/** Key a plan answer is cached under, or null when it should not be cached (cache off, or a notable replan). */
export function planCacheKey(
  cfg: LifeConfig,
  spec: WorldSpec,
  npcId: string,
  mind: NpcMind | undefined,
  clock: number,
  place: string,
): string | null {
  return cfg.cache && !mind?.replan
    ? planKey(spec, npcId, mind, clock, place)
    : null;
}

function relDay<T extends { start?: number }>(xs: T[], day0: number): T[] {
  return xs.map((x) =>
    typeof x.start === "number" ? { ...x, start: x.start - day0 } : x,
  );
}

function relOptions(os: PlanOption[], day0: number): PlanOption[] {
  return os.map((o) =>
    o.until.type === "time"
      ? { ...o, until: { ...o.until, at: o.until.at - day0 } }
      : o,
  );
}

export function cachePut(
  life: LifeState | undefined,
  size: number,
  entry: CacheEntry,
): LifeState {
  const cache = (life?.cache ?? []).filter((e) => e.key !== entry.key);
  cache.push(entry);
  return { cache: cache.slice(Math.max(0, cache.length - size)) };
}

export function storePlan(
  st: WorldState,
  cfg: LifeConfig,
  key: string | null,
  out: unknown,
  clock: number,
): WorldState {
  if (!cfg.cache || !key) return st;
  const o = out as { plan?: PlanBlock[]; options?: unknown } | undefined;
  const plan = Array.isArray(o?.plan) ? o.plan : [];
  const options = parseOptions(o?.options);
  if (!plan.length && !options?.length) return st;
  const day0 = dayOf(clock) * 1440;
  return {
    ...st,
    life: cachePut(st.life, cfg.cache, {
      key,
      plan: relDay(plan, day0),
      ...(options ? { options: relOptions(options, day0) } : {}),
    }),
  };
}

const jitter = (rnd: () => number, span: number) =>
  Math.round(((rnd() * 2 - 1) * span) / 5) * 5;

/** A cached plan made personal: seeded shifts of times and thresholds, past steps dropped. */
export function vary(
  e: CacheEntry,
  seed: number,
  npcId: string,
  clock: number,
): { plan: PlanBlock[]; options?: PlanOption[] } | null {
  const day0 = dayOf(clock) * 1440;
  const rnd = seeded(seed, npcId, day0, e.key);
  const plan = e.plan.map((b) => ({
    ...b,
    start: b.start + day0 + jitter(rnd, 15),
  }));
  if (!e.options) return plan.length ? { plan } : null;
  let last = clock;
  const options: PlanOption[] = [];
  for (const o of e.options) {
    let until = o.until;
    if (until.type === "time") {
      const at = until.at + day0 + jitter(rnd, 20);
      if (at <= last) continue;
      last = at;
      until = { type: "time", at };
    } else if (until.type === "need")
      until = {
        ...until,
        above: Math.max(0, Math.min(100, until.above + jitter(rnd, 5))),
      };
    options.push({ ...o, until });
  }
  return options.length ? { plan, options } : null;
}

export function planNote(
  spec: WorldSpec,
  blocks: PlanBlock[],
  day: number,
): Note[] {
  const names = new Map((simOf(spec)?.places ?? []).map((p) => [p.id, p.name]));
  const text = blocks
    .map(
      (b) =>
        `${clockLabel(b.start).split(" · ")[1]} ${b.activity} at ${names.get(b.place) ?? b.place}`,
    )
    .join("; ");
  return text
    ? [{ text: `My plan for day ${day + 1}: ${text}`, importance: 4 }]
    : [];
}

/** A state updater when the cache already holds a plan for this situation, else null (call the model). */
export function cachedPlan(
  spec: WorldSpec,
  cfg: LifeConfig,
  s: WorldState,
  npcId: string,
  clock: number,
  place: string,
): ((st: WorldState) => WorldState) | null {
  const mind = s.npc[npcId];
  const key = planCacheKey(cfg, spec, npcId, mind, clock, place);
  const hit = key ? s.life?.cache.find((e) => e.key === key) : undefined;
  if (!key || !hit) return null;
  const got = vary(hit, spec.seed, npcId, clock);
  if (!got) return null;
  const day = dayOf(clock);
  return (st) => {
    const m = st.npc[npcId] ?? emptyMind();
    return {
      ...st,
      life: cachePut(st.life, cfg.cache, hit),
      npc: {
        ...st.npc,
        [npcId]: remember(
          {
            ...m,
            plan: got.plan,
            planDay: day,
            ...lifePlanned(
              m,
              cfg.options && got.options ? { options: got.options } : null,
              clock,
            ),
          },
          planNote(spec, got.plan, day),
          st.clock,
          "plan",
        ),
      },
    };
  };
}

function pendingRatings(s: WorldState, n: number): [string, MemoryRecord][] {
  const out: [string, MemoryRecord][] = [];
  for (const [id, m] of Object.entries(s.npc))
    for (const r of m.memory) {
      if (out.length >= n) return out;
      if (r.guess) out.push([id, r]);
    }
  return out;
}

type Call = (
  state: Record<string, unknown>,
  signal: AbortSignal,
) => Promise<unknown>;
type Update = (fn: (st: WorldState) => WorldState) => void;

export function lifeJobs(
  spec: WorldSpec,
  cfg: LifeConfig,
  s: WorldState,
  call: Call,
  update: Update,
): SimJob[] {
  if (!cfg.reflectAt) return [];
  const jobs: SimJob[] = [];
  const pending = pendingRatings(s, cfg.rateBatch);
  if (pending.length >= cfg.rateBatch)
    jobs.push({
      key: "rate",
      kind: "rate",
      run: async (signal) => {
        const out = (await call(
          { task: "rate", world: spec, texts: pending.map(([, r]) => r.text) },
          signal,
        )) as { ratings?: number[] } | undefined;
        if (!signal.aborted)
          update((st) => applyRatings(st, pending, out?.ratings ?? []));
      },
    });
  for (const [id, m] of Object.entries(s.npc)) {
    if ((m.since ?? 0) < cfg.reflectAt) continue;
    jobs.push({
      key: `reflect:${id}`,
      kind: "reflect",
      run: async (signal) => {
        // A failed call keeps the importance sum, so the reflection is tried again.
        const out = (await call(
          {
            task: "reflect",
            world: spec,
            npc_id: id,
            memory: m.memory.filter((r) => !r.guess).slice(-30),
          },
          signal,
        )) as { insights?: Note[] } | undefined;
        if (signal.aborted) return;
        const insights = (out?.insights ?? []).filter(
          (x) => typeof x?.text === "string",
        );
        update((st) => {
          const mind = st.npc[id];
          if (!mind) return st;
          const next = remember(mind, insights, st.clock, "reflection");
          // Only what this reflection saw is spent; notes added while it ran wait for the next one.
          const since = Math.max(0, (mind.since ?? 0) - (m.since ?? 0));
          return { ...st, npc: { ...st.npc, [id]: { ...next, since } } };
        });
      },
    });
  }
  return jobs;
}

export function applyRatings(
  st: WorldState,
  rated: [string, MemoryRecord][],
  ratings: number[],
): WorldState {
  const npc = { ...st.npc };
  rated.forEach(([id, rec], i) => {
    const m = npc[id];
    if (!m) return;
    const r = ratings[i];
    const importance =
      typeof r === "number" && Number.isFinite(r)
        ? Math.max(1, Math.min(10, Math.round(r)))
        : rec.importance;
    let hit = false;
    const memory = m.memory.map((x) => {
      if (x.id !== rec.id || !x.guess) return x;
      hit = true;
      const { guess: _, ...kept } = x;
      return { ...kept, importance };
    });
    if (hit)
      npc[id] = {
        ...m,
        memory,
        ...(m.since !== undefined ? { since: m.since + importance } : {}),
        ...(m.options && importance >= 8
          ? { replan: true, replanSeq: (m.replanSeq ?? 0) + 1 }
          : {}),
      };
  });
  return { ...st, npc };
}

export function pairOf(mind: NpcMind | undefined, other: string): PairMemory {
  return mind?.rel?.[other] ?? { affinity: 0, facts: [] };
}

export function feel(
  mind: NpcMind,
  other: string,
  delta: number,
  fact?: string,
): NpcMind {
  if (!mind.rel) return mind;
  const p = pairOf(mind, other);
  const f = fact?.trim().slice(0, 80);
  const facts = f
    ? [...p.facts.filter((x) => x !== f), f].slice(-MAX_FACTS)
    : p.facts;
  const affinity = Math.max(
    -10,
    Math.min(10, p.affinity + (Number.isFinite(delta) ? delta : 0)),
  );
  return { ...mind, rel: { ...mind.rel, [other]: { ...p, affinity, facts } } };
}

/** Notes the listener heard from `speaker`, each carrying who passed it on (nearest first). */
export function rumours(
  s: WorldState,
  speaker: string,
  notes: (Note & { from?: string })[],
): Note[] {
  const said = s.npc[speaker]?.memory ?? [];
  return notes.map(({ from, ...n }) => {
    const prior = from ? (said.find((r) => r.id === from)?.source ?? []) : [];
    return {
      ...n,
      source: [speaker, ...prior.filter((x) => x !== speaker)].slice(
        0,
        MAX_CHAIN,
      ),
    };
  });
}

/** "Who told you?": names along a memory's source chain, nearest first. */
export function whoToldMe(
  spec: WorldSpec,
  s: WorldState,
  npcId: string,
  recordId: string,
): string[] {
  const rec = s.npc[npcId]?.memory.find((r) => r.id === recordId);
  const name = (id: string) =>
    id === "player"
      ? spec.player.name
      : (spec.npcs.find((n) => n.id === id)?.name ?? id);
  return (rec?.source ?? []).map(name);
}

export function chatAsk(cfg: LifeConfig, s: WorldState, a: string, b: string) {
  return cfg.relationships
    ? { rel_a: pairOf(s.npc[a], b), rel_b: pairOf(s.npc[b], a) }
    : {};
}

interface Feels {
  affinity?: number;
  fact?: string;
}

export function afterChat(
  cfg: LifeConfig,
  st: WorldState,
  a: string,
  b: string,
  c: { a_feels?: Feels; b_feels?: Feels },
): WorldState {
  if (!cfg.relationships) return st;
  const npc = { ...st.npc };
  const talked = (m: NpcMind, other: string): NpcMind =>
    m.rel
      ? {
          ...m,
          rel: { ...m.rel, [other]: { ...pairOf(m, other), talked: st.clock } },
        }
      : m;
  if (npc[a])
    npc[a] = talked(
      feel(npc[a], b, c.a_feels?.affinity ?? 0, c.a_feels?.fact),
      b,
    );
  if (npc[b])
    npc[b] = talked(
      feel(npc[b], a, c.b_feels?.affinity ?? 0, c.b_feels?.fact),
      a,
    );
  return { ...st, npc };
}

export function recordGift(
  cfg: LifeConfig,
  st: WorldState,
  to: string,
  from: string,
  what: string,
): WorldState {
  const m = st.npc[to];
  if (!cfg.relationships || !m) return st;
  return {
    ...st,
    npc: { ...st.npc, [to]: feel(m, from, 2, `gave me ${what}`) },
  };
}

const SMALL_TALK = [
  ["Morning! Busy day?", "Same as ever. You?"],
  ["Nice weather for it.", "Can't complain."],
  ["Good to see you.", "Likewise, take care."],
  ["How's things?", "Ticking along, thanks."],
];

function notableSince(m: NpcMind | undefined, clock: number): boolean {
  return !!m?.memory.some(
    (r) =>
      r.kind === "observation" &&
      r.importance >= NOTABLE &&
      r.t_created * 60 > clock,
  );
}

/** After a pair's first model chat of the day they small-talk in code, unless either has notable news. */
export function smallTalk(opts: {
  cfg: LifeConfig;
  spec: WorldSpec;
  s: WorldState;
  a: string;
  b: string;
  clock: number;
  say: (npcId: string, text: string, ms: number) => void;
  update: Update;
}): boolean {
  const { cfg, spec, s, a, b, clock, say, update } = opts;
  if (!cfg.relationships) return false;
  const last = pairOf(s.npc[a], b).talked ?? pairOf(s.npc[b], a).talked;
  if (
    last === undefined ||
    dayOf(last) !== dayOf(clock) ||
    notableSince(s.npc[a], last) ||
    notableSince(s.npc[b], last)
  )
    return false;
  const rnd = seeded(spec.seed, a, b, Math.floor(clock));
  const fact = pairOf(s.npc[a], b).facts.at(-1);
  const pair = SMALL_TALK[Math.floor(rnd() * SMALL_TALK.length)];
  const first =
    fact && rnd() < 0.5 ? `Still thinking about how ${fact}.` : pair[0];
  say(a, first, 3000);
  setTimeout(() => say(b, pair[1], 3000), 1500);
  update((st) => {
    const npc = { ...st.npc };
    if (npc[a]) npc[a] = feel(npc[a], b, rnd() < 0.3 ? 1 : 0);
    if (npc[b]) npc[b] = feel(npc[b], a, rnd() < 0.3 ? 1 : 0);
    return { ...st, npc };
  });
  return true;
}

export function lifeMindOf(
  raw: unknown,
  fresh: NpcMind,
  optionsOn = true,
): Partial<NpcMind> {
  const r = (raw ?? {}) as Partial<NpcMind>;
  const out: Partial<NpcMind> = {};
  // Option plans only advance while option planning runs, so drop them when it is off.
  const options = optionsOn ? parseOptions(r.options) : undefined;
  if (options) out.options = options;
  const keys = optionsOn
    ? (["optionAt", "optionSince", "planAt", "since"] as const)
    : (["planAt", "since"] as const);
  for (const k of keys)
    if (typeof r[k] === "number" && Number.isFinite(r[k])) out[k] = r[k];
  if (optionsOn && r.replan === true) out.replan = true;
  if (fresh.since !== undefined && out.since === undefined) out.since = 0;
  if (fresh.since === undefined) delete out.since;
  if (fresh.rel) {
    const rel: Record<string, PairMemory> = {};
    for (const [id, p] of Object.entries(
      (r.rel ?? {}) as Record<string, PairMemory>,
    ))
      if (p && typeof p.affinity === "number" && Array.isArray(p.facts))
        rel[id] = {
          affinity: Math.max(-10, Math.min(10, p.affinity)),
          facts: p.facts.filter((f) => typeof f === "string").slice(-MAX_FACTS),
          ...(typeof p.talked === "number" ? { talked: p.talked } : {}),
        };
    out.rel = rel;
  }
  return out;
}

export function lifeStateOf(
  raw: unknown,
  cfg: LifeConfig,
): LifeState | undefined {
  if (!cfg.cache) return undefined;
  const cache: CacheEntry[] = [];
  for (const e of ((raw as LifeState | null)?.cache ?? []) as CacheEntry[]) {
    if (!e || typeof e.key !== "string" || !Array.isArray(e.plan)) continue;
    const options =
      e.options === undefined ? undefined : parseOptions(e.options);
    cache.push({
      key: e.key,
      plan: e.plan.filter(
        (b) => b && typeof b.start === "number" && typeof b.place === "string",
      ),
      ...(options ? { options } : {}),
    });
  }
  return { cache: cache.slice(-cfg.cache) };
}
