// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Village Rivals: a turn-based 4X-lite on the generated map. The LLM only sets each AI faction's posture once a
// turn; deterministic managers (build queue, workers, raid squads, diplomacy) execute it.

import type { WorldFaction, WorldSpec } from "../../types";
import { type Good, makesOf } from "./autonomy";
import { seeded } from "./needs";
import { type WorldMap, isSolid } from "./procgen";

export const RESOURCES = ["food", "wood", "stone", "gold"] as const;
export type Res = (typeof RESOURCES)[number];
export type Stock = Record<Res, number>;
export const POSTURE_KEYS = [
  "economy",
  "expansion",
  "military",
  "diplomacy",
] as const;
export type PostureKey = (typeof POSTURE_KEYS)[number];
export type Posture = Record<PostureKey, number>;
export type Relation = "war" | "peace" | "ally";

export const BUILDINGS = {
  farm: { name: "Farm", cost: { wood: 2 }, turns: 1, job: "food" },
  lumber: { name: "Lumber camp", cost: { food: 2 }, turns: 1, job: "wood" },
  quarry: {
    name: "Quarry",
    cost: { wood: 2, food: 1 },
    turns: 1,
    job: "stone",
  },
  market: {
    name: "Market",
    cost: { wood: 2, stone: 2 },
    turns: 2,
    job: "gold",
  },
  house: { name: "House", cost: { wood: 2, stone: 1 }, turns: 1, job: null },
  barracks: {
    name: "Barracks",
    cost: { wood: 2, stone: 2 },
    turns: 2,
    job: null,
  },
} as const satisfies Record<
  string,
  { name: string; cost: Partial<Stock>; turns: number; job: Res | null }
>;
export type Building = keyof typeof BUILDINGS;
export const BUILDING_IDS = Object.keys(BUILDINGS) as Building[];

export const PLOT = 6;
export const ACTIONS_PER_TURN = 2;
export const MAX_QUEUE = 4;
export const MAX_LOG = 40;
export const MAX_OFFERS = 3;
export const DIGEST_LOG = 6;
export const DIGEST_MAX_CHARS = 2400;
export const HYSTERESIS = 0.15;
export const WIN_SHARE = 0.6;
const BASE_OUT: Stock = { food: 2, wood: 2, stone: 1, gold: 1 };
const CLAIM_COST: Partial<Stock> = { food: 2, wood: 1 };
const TRAIN_COST: Partial<Stock> = { food: 2, gold: 1 };
const START: Stock = { food: 8, wood: 6, stone: 3, gold: 3 };
const MIN_WANT = 8;
export const STOCK_CAP = 60;
const GOOD_JOB: Record<Good, Res> = {
  bread: "food",
  fish: "food",
  apples: "food",
  wood: "wood",
  stone: "stone",
  tools: "gold",
  flowers: "gold",
};

export interface QueuedBuild {
  kind: Building;
  left: number;
  want: number;
}

export interface FactionState {
  stock: Stock;
  workers: number;
  soldiers: number;
  buildings: Record<Building, number>;
  queue: QueuedBuild[];
  posture: Posture;
  target: string | null;
  goals: string[];
  jobs: Stock;
  villagers: string[];
  actionsLeft: number;
  out?: boolean;
}

export interface RivalOffer {
  id: number;
  from: string;
  to: string;
  action: RivalAction;
  text: string;
}

export interface RivalsState {
  turn: number;
  factions: Record<string, FactionState>;
  plots: Record<string, string>;
  relations: Record<string, Relation>;
  log: { turn: number; text: string }[];
  offers: RivalOffer[];
  nextOffer: number;
  over: { winner: string | null; reason: string } | null;
}

export type RivalAction =
  | { kind: "build"; building: Building }
  | { kind: "claim"; plot?: string }
  | { kind: "train" }
  | { kind: "raid"; target: string }
  | {
      kind: "trade";
      with: string;
      give: Res;
      giveAmount: number;
      want: Res;
      wantAmount: number;
    }
  | { kind: "ally"; with: string }
  | { kind: "peace"; with: string };

export interface StrategistPlan {
  posture: Posture;
  target: string | null;
  goals: string[];
  say?: string;
}

export interface Board {
  cols: number;
  rows: number;
  land: Uint8Array;
  home: Record<string, string[]>;
  factions: WorldFaction[];
  player: string | null;
  seed: number;
}

export const rivalsOn = (spec: WorldSpec) =>
  (spec.rivals?.factions?.length ?? 0) >= 2;

export const plotKey = (px: number, py: number) => `${px},${py}`;
const parseKey = (k: string) => k.split(",").map(Number) as [number, number];
export const relKey = (a: string, b: string) =>
  a < b ? `${a}|${b}` : `${b}|${a}`;
export const relationOf = (s: RivalsState, a: string, b: string): Relation =>
  s.relations[relKey(a, b)] ?? "peace";

export const BALANCED: Posture = {
  economy: 0.25,
  expansion: 0.25,
  military: 0.25,
  diplomacy: 0.25,
};

export function normPosture(raw: Partial<Record<string, unknown>>): Posture {
  const w = POSTURE_KEYS.map((k) => {
    const v = Number(raw?.[k]);
    return Number.isFinite(v) && v > 0 ? Math.min(v, 1e6) : 0;
  });
  const sum = w.reduce((a, b) => a + b, 0);
  if (sum <= 0) return { ...BALANCED };
  const out = {} as Posture;
  POSTURE_KEYS.forEach((k, i) => {
    out[k] = Math.round((w[i] / sum) * 1000) / 1000;
  });
  return out;
}

/** Half the L1 distance between two postures: 0 = same, 1 = opposite. */
export const postureDelta = (a: Posture, b: Posture) =>
  POSTURE_KEYS.reduce((s, k) => s + Math.abs(a[k] - b[k]), 0) / 2;

export function boardOf(spec: WorldSpec, map: WorldMap): Board {
  const factions = (spec.rivals?.factions ?? []).slice(0, 4);
  const cols = Math.ceil(map.width / PLOT);
  const rows = Math.ceil(map.height / PLOT);
  const land = new Uint8Array(cols * rows);
  for (let py = 0; py < rows; py++)
    for (let px = 0; px < cols; px++) {
      let open = 0;
      let all = 0;
      for (let y = py * PLOT; y < Math.min(map.height, (py + 1) * PLOT); y++)
        for (let x = px * PLOT; x < Math.min(map.width, (px + 1) * PLOT); x++) {
          all++;
          if (!isSolid(map, x, y)) open++;
        }
      land[py * cols + px] = open * 2 >= all ? 1 : 0;
    }
  const taken = new Set<string>();
  const home: Record<string, string[]> = {};
  const corners: [number, number][] = [
    [0, 0],
    [cols - 1, rows - 1],
    [cols - 1, 0],
    [0, rows - 1],
  ];
  factions.forEach((f, i) => {
    const r = spec.regions.find((x) => x.id === f.region);
    let keys: string[] = [];
    if (r) {
      const x0 = Math.floor(r.x / PLOT);
      const y0 = Math.floor(r.y / PLOT);
      const x1 = Math.floor((r.x + Math.max(1, r.w) - 1) / PLOT);
      const y1 = Math.floor((r.y + Math.max(1, r.h) - 1) / PLOT);
      for (let py = y0; py <= Math.min(y1, rows - 1); py++)
        for (let px = x0; px <= Math.min(x1, cols - 1); px++)
          keys.push(plotKey(px, py));
    }
    keys = keys.filter((k) => !taken.has(k)).slice(0, 2);
    if (!keys.length) {
      const [cx, cy] = corners[i % corners.length];
      const near = [...Array(cols * rows).keys()]
        .map((n) => [n % cols, Math.floor(n / cols)] as [number, number])
        .filter(([px, py]) => !taken.has(plotKey(px, py)))
        .sort(
          (a, b) =>
            Math.abs(a[0] - cx) +
              Math.abs(a[1] - cy) -
              (Math.abs(b[0] - cx) + Math.abs(b[1] - cy)) ||
            land[b[1] * cols + b[0]] - land[a[1] * cols + a[0]],
        );
      if (near[0]) keys = [plotKey(near[0][0], near[0][1])];
    }
    for (const k of keys) {
      taken.add(k);
      const [px, py] = parseKey(k);
      land[py * cols + px] = 1;
    }
    home[f.id] = keys;
  });
  return {
    cols,
    rows,
    land,
    home,
    factions,
    player: factions.find((f) => f.player)?.id ?? null,
    seed: spec.seed,
  };
}

const emptyStock = (): Stock => ({ food: 0, wood: 0, stone: 0, gold: 0 });
const noBuildings = () =>
  Object.fromEntries(BUILDING_IDS.map((b) => [b, 0])) as Record<
    Building,
    number
  >;

function villagersOf(spec: WorldSpec, map: WorldMap, b: Board, fid: string) {
  const f = b.factions.find((x) => x.id === fid);
  const mine = new Set(b.home[fid]);
  const out = spec.npcs
    .filter((n) => {
      if (n.id === f?.leader) return true;
      const p = map.npcs[n.id];
      if (!p) return false;
      const owner = b.factions.find(
        (o) =>
          o.leader !== n.id &&
          b.home[o.id]?.includes(
            plotKey(Math.floor(p.x / PLOT), Math.floor(p.y / PLOT)),
          ),
      );
      return owner?.id === fid && mine.size > 0;
    })
    .map((n) => n.id);
  return [...new Set(out)].slice(0, 6);
}

function startPosture(seed: number, fid: string, player: boolean): Posture {
  if (player) return { ...BALANCED };
  const rnd = seeded(seed, fid, "posture");
  return normPosture({
    economy: 1 + rnd(),
    expansion: 1 + rnd(),
    military: 0.6 + rnd(),
    diplomacy: 0.6 + rnd(),
  });
}

export function initialRivals(
  spec: WorldSpec,
  map: WorldMap,
  board = boardOf(spec, map),
): RivalsState {
  const factions: Record<string, FactionState> = {};
  const plots: Record<string, string> = {};
  for (const f of board.factions) {
    const villagers = villagersOf(spec, map, board, f.id);
    factions[f.id] = {
      stock: { ...START },
      workers: 3 + Math.min(3, villagers.length),
      soldiers: 1,
      buildings: noBuildings(),
      queue: [],
      posture: startPosture(spec.seed, f.id, Boolean(f.player)),
      target: null,
      goals: [],
      jobs: emptyStock(),
      villagers,
      actionsLeft: ACTIONS_PER_TURN,
    };
    for (const k of board.home[f.id] ?? []) plots[k] = f.id;
  }
  let s: RivalsState = {
    turn: 1,
    factions,
    plots,
    relations: {},
    log: [{ turn: 1, text: "The villages eye each other. Day 1 begins." }],
    offers: [],
    nextOffer: 1,
    over: null,
  };
  for (const f of board.factions) s = assignWorkers(spec, s, f.id);
  return s;
}

const nameOf = (b: Board, id: string) =>
  b.factions.find((f) => f.id === id)?.name ?? id;

export function plotsOf(s: RivalsState, fid: string): string[] {
  return Object.keys(s.plots)
    .filter((k) => s.plots[k] === fid)
    .sort();
}

export function landPlots(b: Board): number {
  return b.land.reduce((a, x) => a + x, 0);
}

export function scoreOf(s: RivalsState, fid: string): number {
  const f = s.factions[fid];
  if (!f) return 0;
  const built = BUILDING_IDS.reduce((a, k) => a + f.buildings[k], 0);
  const stock = RESOURCES.reduce((a, r) => a + f.stock[r], 0);
  return (
    plotsOf(s, fid).length * 3 +
    built * 2 +
    f.workers +
    f.soldiers +
    Math.floor(stock / 10)
  );
}

const canPay = (stock: Stock, cost: Partial<Stock>) =>
  RESOURCES.every((r) => stock[r] >= (cost[r] ?? 0));
const pay = (stock: Stock, cost: Partial<Stock>): Stock => {
  const out = { ...stock };
  for (const r of RESOURCES) out[r] -= cost[r] ?? 0;
  return out;
};

export function frontier(b: Board, s: RivalsState, fid: string): string[] {
  const mine = plotsOf(s, fid);
  const seen = new Set<string>();
  for (const k of mine) {
    const [px, py] = parseKey(k);
    for (const [dx, dy] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]) {
      const nx = px + dx;
      const ny = py + dy;
      if (nx < 0 || ny < 0 || nx >= b.cols || ny >= b.rows) continue;
      const nk = plotKey(nx, ny);
      if (!s.plots[nk] && b.land[ny * b.cols + nx]) seen.add(nk);
    }
  }
  const home = b.home[fid]?.[0];
  const [hx, hy] = home ? parseKey(home) : [0, 0];
  return [...seen].sort((a, c) => {
    const [ax, ay] = parseKey(a);
    const [cx, cy] = parseKey(c);
    return (
      Math.abs(ax - hx) +
        Math.abs(ay - hy) -
        (Math.abs(cx - hx) + Math.abs(cy - hy)) || (a < c ? -1 : 1)
    );
  });
}

const upkeep = (f: FactionState) => Math.ceil((f.workers + f.soldiers) / 2);
const yieldOf = (f: FactionState, r: Res) => {
  const site = BUILDING_IDS.find((k) => BUILDINGS[k].job === r);
  return BASE_OUT[r] * (site ? 1 + 0.5 * f.buildings[site] : 1);
};
const soldierCap = (f: FactionState) => 2 + 3 * f.buildings.barracks;
const popCap = (s: RivalsState, fid: string) =>
  4 + 2 * s.factions[fid].buildings.house + plotsOf(s, fid).length;

function addLog(s: RivalsState, text: string): RivalsState {
  return {
    ...s,
    log: [...s.log, { turn: s.turn, text: text.slice(0, 160) }].slice(-MAX_LOG),
  };
}

function setFaction(s: RivalsState, fid: string, f: FactionState): RivalsState {
  return { ...s, factions: { ...s.factions, [fid]: f } };
}

export interface ActResult {
  state: RivalsState;
  ok: boolean;
  why?: string;
}

/** Why an action cannot run now, or null when it can. The one gate for the player, the managers and offers. */
export function blocked(
  b: Board,
  s: RivalsState,
  fid: string,
  a: RivalAction,
): string | null {
  const f = s.factions[fid];
  if (!f || f.out) return "that village is out of the game";
  if (s.over) return "the game is over";
  if (f.actionsLeft <= 0) return "no actions left this day";
  const other = "with" in a ? a.with : a.kind === "raid" ? a.target : null;
  if (other !== null) {
    if (other === fid || !s.factions[other] || s.factions[other].out)
      return "unknown village";
  }
  switch (a.kind) {
    case "build":
      if (!BUILDINGS[a.building]) return "unknown building";
      if (f.queue.length >= MAX_QUEUE) return "the build queue is full";
      return canPay(f.stock, BUILDINGS[a.building].cost)
        ? null
        : "not enough materials";
    case "claim": {
      const free = frontier(b, s, fid);
      if (!free.length) return "no free land next to your plots";
      if (a.plot && !free.includes(a.plot)) return "that plot is not free";
      return canPay(f.stock, CLAIM_COST) ? null : "not enough food and wood";
    }
    case "train":
      if (f.workers <= 1) return "no worker to spare";
      if (f.soldiers >= soldierCap(f))
        return "build barracks for more soldiers";
      return canPay(f.stock, TRAIN_COST) ? null : "not enough food and gold";
    case "raid":
      if (relationOf(s, fid, a.target) === "ally") return "you are allies";
      return f.soldiers >= 1 ? null : "no soldiers";
    case "trade": {
      const o = s.factions[a.with];
      if (relationOf(s, fid, a.with) === "war") return "you are at war";
      if (a.give === a.want) return "a trade needs two goods";
      const ok = (n: number) => Number.isInteger(n) && n >= 1 && n <= 10;
      if (!ok(a.giveAmount) || !ok(a.wantAmount)) return "bad amounts";
      if (!RESOURCES.includes(a.give) || !RESOURCES.includes(a.want))
        return "unknown goods";
      if (f.stock[a.give] < a.giveAmount) return `not enough ${a.give}`;
      if (o.stock[a.want] < a.wantAmount) return `${a.with} lacks ${a.want}`;
      return null;
    }
    case "ally":
      return relationOf(s, fid, a.with) === "peace"
        ? null
        : "an alliance needs peace first";
    case "peace":
      return relationOf(s, fid, a.with) === "war" ? null : "you are not at war";
  }
  return "unknown action";
}

function raid(b: Board, s: RivalsState, fid: string, target: string) {
  const a = s.factions[fid];
  const d = s.factions[target];
  const share = 0.5 + 0.5 * a.posture.military;
  const squad = Math.max(1, Math.round(a.soldiers * share));
  const rnd = seeded(b.seed, "raid", s.turn, fid, target);
  const atk = squad * (1 + 0.2 * a.buildings.barracks) * (0.8 + 0.4 * rnd());
  const def = d.soldiers * (1 + 0.1 * d.buildings.barracks) + 1;
  let next: RivalsState = {
    ...s,
    relations: { ...s.relations, [relKey(fid, target)]: "war" },
  };
  if (atk > def) {
    const loot = emptyStock();
    for (const r of RESOURCES) loot[r] = Math.floor(d.stock[r] * 0.3);
    const theirs = plotsOf(next, target);
    const mine = new Set(plotsOf(next, fid));
    const edge = theirs.filter((k) => {
      const [px, py] = parseKey(k);
      return [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ].some(([dx, dy]) => mine.has(plotKey(px + dx, py + dy)));
    });
    const plots = { ...next.plots };
    const taken = theirs.length > 1 && edge.length ? edge[0] : null;
    if (taken) plots[taken] = fid;
    const lostD = Math.min(d.soldiers, Math.ceil(d.soldiers * 0.3));
    const lostA = Math.floor(squad * 0.2);
    next = {
      ...next,
      plots,
      factions: {
        ...next.factions,
        [fid]: {
          ...a,
          soldiers: a.soldiers - lostA,
          stock: Object.fromEntries(
            RESOURCES.map((r) => [r, a.stock[r] + loot[r]]),
          ) as Stock,
        },
        [target]: {
          ...d,
          soldiers: d.soldiers - lostD,
          stock: Object.fromEntries(
            RESOURCES.map((r) => [r, d.stock[r] - loot[r]]),
          ) as Stock,
        },
      },
    };
    return addLog(
      next,
      `${nameOf(b, fid)} raided ${nameOf(b, target)} and won: ${RESOURCES.map((r) => `${loot[r]} ${r}`).join(", ")}${taken ? ", and a plot" : ""}.`,
    );
  }
  const lost = Math.min(a.soldiers, Math.ceil(squad * 0.5));
  next = setFaction(next, fid, { ...a, soldiers: a.soldiers - lost });
  return addLog(
    next,
    `${nameOf(b, fid)} raided ${nameOf(b, target)} and was driven off (lost ${lost} soldiers).`,
  );
}

/** Run one action for a village; every caller (player, managers, accepted offers) goes through here. */
export function act(
  b: Board,
  s: RivalsState,
  fid: string,
  a: RivalAction,
  want = 50,
): ActResult {
  const why = blocked(b, s, fid, a);
  if (why) return { state: s, ok: false, why };
  let f: FactionState = {
    ...s.factions[fid],
    actionsLeft: s.factions[fid].actionsLeft - 1,
  };
  let next = s;
  const me = nameOf(b, fid);
  switch (a.kind) {
    case "build": {
      const spec = BUILDINGS[a.building];
      f = {
        ...f,
        stock: pay(f.stock, spec.cost),
        queue: [...f.queue, { kind: a.building, left: spec.turns, want }].sort(
          (x, y) => y.want - x.want,
        ),
      };
      next = addLog(
        setFaction(next, fid, f),
        `${me} started a ${spec.name.toLowerCase()}.`,
      );
      break;
    }
    case "claim": {
      const plot = a.plot ?? frontier(b, s, fid)[0];
      f = { ...f, stock: pay(f.stock, CLAIM_COST) };
      next = setFaction(
        { ...next, plots: { ...next.plots, [plot]: fid } },
        fid,
        f,
      );
      next = addLog(next, `${me} claimed new land.`);
      break;
    }
    case "train":
      f = {
        ...f,
        stock: pay(f.stock, TRAIN_COST),
        workers: f.workers - 1,
        soldiers: f.soldiers + 1,
      };
      next = addLog(setFaction(next, fid, f), `${me} trained a soldier.`);
      break;
    case "raid":
      next = raid(b, setFaction(next, fid, f), fid, a.target);
      break;
    case "trade": {
      const o = s.factions[a.with];
      f = {
        ...f,
        stock: {
          ...f.stock,
          [a.give]: f.stock[a.give] - a.giveAmount,
          [a.want]: f.stock[a.want] + a.wantAmount,
        },
      };
      const g = {
        ...o,
        stock: {
          ...o.stock,
          [a.give]: o.stock[a.give] + a.giveAmount,
          [a.want]: o.stock[a.want] - a.wantAmount,
        },
      };
      next = setFaction(setFaction(next, fid, f), a.with, g);
      next = addLog(
        next,
        `${me} traded ${a.giveAmount} ${a.give} to ${nameOf(b, a.with)} for ${a.wantAmount} ${a.want}.`,
      );
      break;
    }
    case "ally":
    case "peace":
      next = setFaction(next, fid, f);
      next = {
        ...next,
        relations: {
          ...next.relations,
          [relKey(fid, a.with)]: a.kind === "ally" ? "ally" : "peace",
        },
      };
      next = addLog(
        next,
        a.kind === "ally"
          ? `${me} and ${nameOf(b, a.with)} are now allies.`
          : `${me} made peace with ${nameOf(b, a.with)}.`,
      );
      break;
  }
  return { state: next, ok: true };
}

/** Hysteresis: a new posture replaces the old one only when it differs by more than `threshold`. */
export function applyPlan(
  b: Board,
  s: RivalsState,
  fid: string,
  plan: StrategistPlan | null,
  threshold = HYSTERESIS,
): RivalsState {
  const f = s.factions[fid];
  if (!plan || !f || f.out) return s;
  const posture = normPosture(plan.posture);
  const targetOk = (t: string | null) =>
    t !== null && t !== fid && !!s.factions[t] && !s.factions[t].out;
  const target = targetOk(plan.target) ? plan.target : null;
  const goals = plan.goals
    .map((g) => String(g).slice(0, 60))
    .filter(Boolean)
    .slice(0, 3);
  const first = f.goals.length === 0;
  const lostTarget = f.target !== null && !targetOk(f.target);
  if (!first && !lostTarget && postureDelta(f.posture, posture) <= threshold)
    return s;
  let next = setFaction(s, fid, {
    ...f,
    posture,
    target,
    goals: goals.length ? goals : ["grow the village"],
  });
  const top = POSTURE_KEYS.reduce((a, k) => (posture[k] > posture[a] ? k : a));
  next = addLog(
    next,
    `${nameOf(b, fid)} now leans to ${top}${target ? `, eyeing ${nameOf(b, target)}` : ""}.${plan.say ? ` "${plan.say.slice(0, 100)}"` : ""}`,
  );
  return next;
}

/** Freeciv-style wants: need x posture weight, on a 0-100 scale. */
export function wants(
  b: Board,
  s: RivalsState,
  fid: string,
): { action: RivalAction; want: number }[] {
  const f = s.factions[fid];
  const p = f.posture;
  const out: { action: RivalAction; want: number }[] = [];
  const pop = f.workers + f.soldiers;
  const short = (r: Res, target: number) =>
    Math.max(0, Math.min(1, (target - f.stock[r]) / target));
  const needs: Stock = {
    food: short("food", 6 + pop),
    wood: short("wood", 6 + 6 * p.expansion),
    stone: short("stone", 4 + 4 * p.military),
    gold: short("gold", 3 + 6 * p.diplomacy),
  };
  const queued = (k: Building) =>
    f.buildings[k] + f.queue.filter((q) => q.kind === k).length;
  for (const k of ["farm", "lumber", "quarry"] as const) {
    const job = BUILDINGS[k].job;
    out.push({
      action: { kind: "build", building: k },
      want: (100 * p.economy * (0.4 + needs[job])) / (1 + queued(k)),
    });
  }
  out.push({
    action: { kind: "build", building: "market" },
    want: (100 * (p.economy + p.diplomacy) * 0.5) / (1 + queued("market")),
  });
  const crowded = pop / Math.max(1, popCap(s, fid));
  out.push({
    action: { kind: "build", building: "house" },
    want:
      (100 * (p.expansion + p.economy * 0.5) * crowded) / (1 + queued("house")),
  });
  out.push({
    action: { kind: "build", building: "barracks" },
    want: (100 * p.military) / (1 + 2 * queued("barracks")),
  });
  if (frontier(b, s, fid).length)
    out.push({ action: { kind: "claim" }, want: 100 * p.expansion });
  out.push({
    action: { kind: "train" },
    want: 100 * p.military * (1 - f.soldiers / Math.max(1, soldierCap(f))),
  });
  const target = f.target && s.factions[f.target] ? f.target : null;
  if (target && relationOf(s, fid, target) !== "ally") {
    const squad = Math.round(f.soldiers * (0.5 + 0.5 * p.military));
    const ratio = squad / (s.factions[target].soldiers + 1);
    if (ratio >= 1.2)
      out.push({
        action: { kind: "raid", target },
        want: 100 * p.military * Math.min(1.5, ratio / 1.5),
      });
  }
  for (const o of b.factions) {
    if (o.id === fid || o.player || s.factions[o.id]?.out) continue;
    const rel = relationOf(s, fid, o.id);
    if (rel === "war" && o.id !== target)
      out.push({
        action: { kind: "peace", with: o.id },
        want: 100 * p.diplomacy,
      });
    if (
      rel === "peace" &&
      o.id !== target &&
      s.factions[o.id].target !== fid &&
      s.factions[o.id].posture.diplomacy >= 0.25
    )
      out.push({
        action: { kind: "ally", with: o.id },
        want: 80 * p.diplomacy,
      });
    const trade = tradeFor(s, fid, o.id);
    if (trade)
      out.push({ action: trade, want: 60 * (p.diplomacy + p.economy) });
  }
  return out
    .map((x) => ({ ...x, want: Math.round(x.want * 10) / 10 }))
    .sort((a, c) => c.want - a.want);
}

/** A 1:1 swap of my most plentiful good for my scarcest, when the partner has that one to spare. */
function tradeFor(
  s: RivalsState,
  fid: string,
  other: string,
): RivalAction | null {
  if (relationOf(s, fid, other) === "war") return null;
  const f = s.factions[fid].stock;
  const o = s.factions[other].stock;
  const most = RESOURCES.reduce((a, r) => (f[r] > f[a] ? r : a));
  const least = RESOURCES.reduce((a, r) => (f[r] < f[a] ? r : a));
  if (most === least || f[most] - f[least] < 6) return null;
  const n = Math.min(3, Math.floor((f[most] - f[least]) / 3));
  if (n < 1 || o[least] < n + 3 || o[most] > o[least]) return null;
  return {
    kind: "trade",
    with: other,
    give: most,
    giveAmount: n,
    want: least,
    wantAmount: n,
  };
}

/** The managers: spend this day's actions on the highest wants the village can afford. */
export function runManagers(
  b: Board,
  s: RivalsState,
  fid: string,
): RivalsState {
  let next = s;
  for (const w of wants(b, s, fid)) {
    if (next.factions[fid].actionsLeft <= 0) break;
    if (w.want < MIN_WANT) break;
    const r = act(b, next, fid, w.action, w.want);
    const f = r.state.factions[fid];
    const farm = w.action.kind === "build" && w.action.building === "farm";
    if (r.ok && (farm || f.stock.food >= 2 * upkeep(f))) next = r.state;
  }
  return proposeToPlayer(b, next, fid);
}

function proposeToPlayer(b: Board, s: RivalsState, fid: string): RivalsState {
  const pid = b.player;
  const f = s.factions[fid];
  if (!pid || s.factions[pid]?.out || f.target === pid) return s;
  if (s.offers.some((o) => o.from === fid) || s.offers.length >= MAX_OFFERS)
    return s;
  const rel = relationOf(s, fid, pid);
  let action: RivalAction | null = null;
  if (rel === "war" && f.posture.diplomacy >= 0.3)
    action = { kind: "peace", with: pid };
  else if (rel === "peace" && f.posture.diplomacy >= 0.4)
    action = { kind: "ally", with: pid };
  if (!action) return s;
  return {
    ...s,
    offers: [
      ...s.offers,
      {
        id: s.nextOffer,
        from: fid,
        to: pid,
        action,
        text:
          action.kind === "peace"
            ? `${nameOf(b, fid)} asks for peace.`
            : `${nameOf(b, fid)} proposes an alliance.`,
      },
    ],
    nextOffer: s.nextOffer + 1,
  };
}

/** Worker assignment: villagers first do what they make (the trade AI's `makesOf`), the rest go to the job
 *  with the highest utility (posture weight x shortage), re-scored after each pick. */
export function assignWorkers(
  spec: WorldSpec,
  s: RivalsState,
  fid: string,
): RivalsState {
  const f = s.factions[fid];
  const jobs = emptyStock();
  const projected = { ...f.stock };
  const eat = upkeep(f) + (f.stock.food < 3 * upkeep(f) ? 1 : 0);
  const crafts = f.villagers
    .slice(0, f.workers)
    .map((id) => GOOD_JOB[makesOf(spec, id)]);
  const fed = () => jobs.food * yieldOf(f, "food") >= eat;
  for (const job of [...crafts].sort(
    (a, c) => (a === "food" ? 0 : 1) - (c === "food" ? 0 : 1),
  )) {
    const j = fed() ? job : "food";
    jobs[j]++;
    projected[j] += BASE_OUT[j];
  }
  let i = crafts.length;
  for (; i < f.workers && !fed(); i++) {
    jobs.food++;
    projected.food += BASE_OUT.food;
  }
  const p = f.posture;
  const pop = f.workers + f.soldiers;
  const weight: Stock = {
    food: 0.5 + p.economy + p.military * 0.5,
    wood: 0.3 + p.economy * 0.5 + p.expansion,
    stone: 0.2 + p.economy * 0.5 + p.military * 0.5,
    gold: 0.1 + p.diplomacy + p.economy * 0.3,
  };
  const target: Stock = {
    food: 6 + 2 * pop,
    wood: 8 + 8 * p.expansion,
    stone: 4 + 6 * p.military,
    gold: 3 + 8 * p.diplomacy,
  };
  for (; i < f.workers; i++) {
    let best: Res = "food";
    let bestU = -1;
    for (const r of RESOURCES) {
      const gap = Math.max(0, target[r] - projected[r]) / target[r];
      const u = weight[r] * (0.05 + gap * gap);
      if (u > bestU) {
        bestU = u;
        best = r;
      }
    }
    jobs[best]++;
    projected[best] += BASE_OUT[best];
  }
  return setFaction(s, fid, { ...f, jobs });
}

function produce(b: Board, s: RivalsState, fid: string): RivalsState {
  let f = s.factions[fid];
  const stock = { ...f.stock };
  for (const r of RESOURCES)
    stock[r] = Math.min(
      STOCK_CAP,
      stock[r] + Math.floor(f.jobs[r] * yieldOf(f, r)),
    );
  stock.food -= upkeep(f);
  let { workers, soldiers } = f;
  let next = s;
  if (stock.food < 0) {
    stock.food = 0;
    if (soldiers > 0) soldiers--;
    else if (workers > 1) workers--;
    next = addLog(next, `${nameOf(b, fid)} went hungry and lost a villager.`);
  } else if (stock.food >= 6 + workers && workers + soldiers < popCap(s, fid)) {
    stock.food -= 3;
    workers++;
  }
  const queue = f.queue.map((q) => ({ ...q }));
  const buildings = { ...f.buildings };
  if (queue.length) {
    queue[0].left--;
    if (queue[0].left <= 0) {
      const done = queue.shift() as QueuedBuild;
      buildings[done.kind]++;
      next = addLog(
        next,
        `${nameOf(b, fid)} finished a ${BUILDINGS[done.kind].name.toLowerCase()}.`,
      );
    }
  }
  f = { ...f, stock, workers, soldiers, queue, buildings };
  return setFaction(next, fid, f);
}

export type Plans = Record<string, StrategistPlan | null | undefined>;

/** One day: plans (with hysteresis), AI managers, then build queues, workers, production and upkeep for all. */
export function endTurn(
  spec: WorldSpec,
  b: Board,
  s: RivalsState,
  plans: Plans,
  threshold = spec.rivals?.hysteresis ?? HYSTERESIS,
): RivalsState {
  if (s.over) return s;
  let next = s;
  for (const f of b.factions)
    if (!f.player)
      next = applyPlan(b, next, f.id, plans[f.id] ?? null, threshold);
  for (const f of b.factions)
    if (!f.player && !next.factions[f.id].out)
      next = runManagers(b, next, f.id);
  for (const f of b.factions) {
    if (next.factions[f.id].out) continue;
    next = assignWorkers(spec, next, f.id);
    next = produce(b, next, f.id);
  }
  for (const f of b.factions) {
    const st = next.factions[f.id];
    if (!st.out && plotsOf(next, f.id).length === 0) {
      next = setFaction(next, f.id, { ...st, out: true });
      next = addLog(next, `${f.name} has lost all its land.`);
    }
  }
  const factions = Object.fromEntries(
    Object.entries(next.factions).map(([id, f]) => [
      id,
      { ...f, actionsLeft: ACTIONS_PER_TURN },
    ]),
  );
  next = {
    ...next,
    factions,
    offers: next.offers.filter((o) => !factions[o.from].out),
    turn: next.turn + 1,
  };
  return { ...next, over: verdict(spec, b, next) };
}

export function verdict(
  spec: WorldSpec,
  b: Board,
  s: RivalsState,
): RivalsState["over"] {
  const alive = b.factions.filter((f) => !s.factions[f.id].out);
  if (alive.length === 1)
    return { winner: alive[0].id, reason: "the last village standing" };
  const land = landPlots(b);
  for (const f of alive)
    if (plotsOf(s, f.id).length >= land * WIN_SHARE)
      return { winner: f.id, reason: "holds most of the land" };
  const max = spec.rivals?.max_turns ?? 30;
  if (s.turn > max) {
    const ranked = [...alive].sort(
      (a, c) => scoreOf(s, c.id) - scoreOf(s, a.id),
    );
    return {
      winner: ranked[0]?.id ?? null,
      reason: `best score after ${max} days`,
    };
  }
  return null;
}

/** The compact state an AI faction's strategist sees each turn; size is bounded whatever the game length. */
export function digest(b: Board, s: RivalsState, fid: string) {
  const f = s.factions[fid];
  const nz = (o: Record<string, number>) =>
    Object.fromEntries(Object.entries(o).filter(([, v]) => v > 0));
  return {
    turn: s.turn,
    max_turns: null as number | null,
    you: {
      id: fid,
      name: nameOf(b, fid).slice(0, 60),
      stock: f.stock,
      workers: f.workers,
      soldiers: f.soldiers,
      plots: plotsOf(s, fid).length,
      buildings: nz(f.buildings),
      queue: f.queue.map((q) => q.kind),
      posture: f.posture,
      target: f.target,
      goals: f.goals,
      free_land: frontier(b, s, fid).length,
    },
    others: b.factions
      .filter((o) => o.id !== fid)
      .map((o) => ({
        id: o.id,
        name: o.name.slice(0, 60),
        player: Boolean(o.player),
        out: Boolean(s.factions[o.id].out),
        plots: plotsOf(s, o.id).length,
        soldiers: s.factions[o.id].soldiers,
        score: scoreOf(s, o.id),
        relation: relationOf(s, fid, o.id),
      })),
    recent: s.log.slice(-DIGEST_LOG).map((l) => l.text.slice(0, 120)),
  };
}

export type Digest = ReturnType<typeof digest>;

export function strategistRequest(
  spec: WorldSpec,
  b: Board,
  s: RivalsState,
  fid: string,
) {
  const d = digest(b, s, fid);
  d.max_turns = spec.rivals?.max_turns ?? 30;
  const f = b.factions.find((x) => x.id === fid);
  const leader = spec.npcs.find((n) => n.id === f?.leader);
  return {
    task: "rivals_strategist",
    world: worldBrief(spec),
    faction: fid,
    leader: leader
      ? { name: leader.name, persona: leader.persona.slice(0, 300) }
      : null,
    digest: d,
  };
}

const worldBrief = (spec: WorldSpec) => ({
  title: spec.title,
  intro_text: spec.intro_text.slice(0, 400),
  rivals: spec.rivals,
});

export function diplomacyRequest(
  spec: WorldSpec,
  b: Board,
  s: RivalsState,
  fid: string,
  message: string,
  history: { role: "player" | "npc"; text: string }[],
) {
  const base = strategistRequest(spec, b, s, fid);
  const f = s.factions[fid];
  return {
    ...base,
    task: "rivals_diplomacy",
    player: b.player,
    intent: { posture: f.posture, target: f.target, goals: f.goals },
    player_message: message.slice(0, 500),
    history: history
      .slice(-6)
      .map((h) => ({ ...h, text: h.text.slice(0, 300) })),
  };
}

/** Parse a strategist reply (already validated by the backend) into a plan, or null. */
export function planOf(raw: unknown): StrategistPlan | null {
  const r = raw as Partial<StrategistPlan> | null;
  if (!r || typeof r !== "object" || !r.posture) return null;
  return {
    posture: normPosture(r.posture as Record<string, unknown>),
    target: typeof r.target === "string" && r.target ? r.target : null,
    goals: Array.isArray(r.goals) ? r.goals.map(String).slice(0, 3) : [],
    ...(typeof r.say === "string" && r.say ? { say: r.say.slice(0, 120) } : {}),
  };
}

export interface DiplomacyOffer {
  kind: "none" | "trade" | "ally" | "peace";
  give?: string;
  give_amount?: number;
  want?: string;
  want_amount?: number;
}

/** A leader's offer as an engine action the player can accept, or null when the engine would refuse it.
 *  `give` is what the leader gives the player. */
export function offerAction(
  b: Board,
  s: RivalsState,
  fid: string,
  offer: DiplomacyOffer | null | undefined,
): RivalAction | null {
  const pid = b.player;
  if (!offer || !pid || offer.kind === "none") return null;
  let a: RivalAction;
  if (offer.kind === "trade") {
    const give = offer.give as Res;
    const want = offer.want as Res;
    a = {
      kind: "trade",
      with: fid,
      give: want,
      giveAmount: Number(offer.want_amount),
      want: give,
      wantAmount: Number(offer.give_amount),
    };
  } else if (offer.kind === "ally" || offer.kind === "peace") {
    if (s.factions[fid]?.target === pid && offer.kind === "ally") return null;
    a = { kind: offer.kind, with: fid };
  } else return null;
  const probe = {
    ...s,
    factions: {
      ...s.factions,
      [pid]: { ...s.factions[pid], actionsLeft: 1 },
    },
  };
  return blocked(b, probe, pid, a) ? null : a;
}

export function describe(b: Board, a: RivalAction): string {
  switch (a.kind) {
    case "build":
      return `Build a ${BUILDINGS[a.building].name.toLowerCase()}`;
    case "claim":
      return "Claim land";
    case "train":
      return "Train a soldier";
    case "raid":
      return `Raid ${nameOf(b, a.target)}`;
    case "trade":
      return `Give ${a.giveAmount} ${a.give} for ${a.wantAmount} ${a.want}`;
    case "ally":
      return `Ally with ${nameOf(b, a.with)}`;
    case "peace":
      return `Make peace with ${nameOf(b, a.with)}`;
  }
}

/** Accept an AI's offer to the player: it runs as the player's action through the same gate. */
export function acceptOffer(b: Board, s: RivalsState, id: number): ActResult {
  const o = s.offers.find((x) => x.id === id);
  if (!o) return { state: s, ok: false, why: "that offer is gone" };
  const rest = { ...s, offers: s.offers.filter((x) => x.id !== id) };
  const r = act(b, rest, o.to, { ...o.action, with: o.from } as RivalAction);
  return r.ok ? r : { ...r, state: rest };
}

export const declineOffer = (s: RivalsState, id: number): RivalsState => ({
  ...s,
  offers: s.offers.filter((x) => x.id !== id),
});

const int = (v: unknown, lo: number, hi: number, d: number) =>
  typeof v === "number" && Number.isFinite(v)
    ? Math.max(lo, Math.min(hi, Math.floor(v)))
    : d;

/** Load a saved game, keeping only what still fits the world; anything unreadable starts fresh. */
export function rivalsOf(
  raw: unknown,
  spec: WorldSpec,
  map: WorldMap,
): RivalsState {
  const b = boardOf(spec, map);
  const fresh = initialRivals(spec, map, b);
  const r = raw as Partial<RivalsState> | null;
  if (!r || typeof r !== "object" || typeof r.turn !== "number") return fresh;
  const ids = new Set(b.factions.map((f) => f.id));
  const factions: Record<string, FactionState> = {};
  for (const id of ids) {
    const x = (r.factions?.[id] ?? null) as Partial<FactionState> | null;
    const base = fresh.factions[id];
    if (!x) {
      factions[id] = base;
      continue;
    }
    const stock = (o: unknown, d: Stock) =>
      Object.fromEntries(
        RESOURCES.map((k) => [k, int((o as Stock | null)?.[k], 0, 1e6, d[k])]),
      ) as Stock;
    factions[id] = {
      ...base,
      stock: stock(x.stock, base.stock),
      jobs: stock(x.jobs, base.jobs),
      workers: int(x.workers, 1, 1000, base.workers),
      soldiers: int(x.soldiers, 0, 1000, base.soldiers),
      buildings: Object.fromEntries(
        BUILDING_IDS.map((k) => [k, int(x.buildings?.[k], 0, 1000, 0)]),
      ) as Record<Building, number>,
      queue: (Array.isArray(x.queue) ? x.queue : [])
        .filter((q) => q && BUILDING_IDS.includes(q.kind))
        .slice(0, MAX_QUEUE)
        .map((q) => ({
          kind: q.kind,
          left: int(q.left, 1, 10, 1),
          want:
            typeof q.want === "number" && Number.isFinite(q.want)
              ? Math.max(0, Math.min(1000, q.want))
              : 0,
        })),
      posture: x.posture ? normPosture(x.posture) : base.posture,
      target:
        typeof x.target === "string" && ids.has(x.target) && x.target !== id
          ? x.target
          : null,
      goals: (Array.isArray(x.goals) ? x.goals : [])
        .map((g) => String(g).slice(0, 60))
        .slice(0, 3),
      actionsLeft: int(x.actionsLeft, 0, ACTIONS_PER_TURN, ACTIONS_PER_TURN),
      ...(x.out ? { out: true } : {}),
    };
  }
  const plots: Record<string, string> = {};
  for (const [k, v] of Object.entries(r.plots ?? {})) {
    const [px, py] = parseKey(k);
    if (
      ids.has(v) &&
      Number.isInteger(px) &&
      Number.isInteger(py) &&
      px >= 0 &&
      py >= 0 &&
      px < b.cols &&
      py < b.rows
    )
      plots[plotKey(px, py)] = v;
  }
  const relations: Record<string, Relation> = {};
  for (const [k, v] of Object.entries(r.relations ?? {})) {
    const [a, c] = k.split("|");
    if (
      ids.has(a) &&
      ids.has(c) &&
      (v === "war" || v === "ally" || v === "peace")
    )
      relations[relKey(a, c)] = v;
  }
  const offers = (Array.isArray(r.offers) ? r.offers : []).filter(
    (o) =>
      o &&
      ids.has(o.from) &&
      ids.has(o.to) &&
      o.action &&
      (o.action.kind === "ally" || o.action.kind === "peace"),
  );
  return {
    turn: int(r.turn, 1, 100000, 1),
    factions,
    plots: Object.keys(plots).length ? plots : fresh.plots,
    relations,
    log: (Array.isArray(r.log) ? r.log : [])
      .filter((l) => l && typeof l.text === "string")
      .map((l) => ({
        turn: int(l.turn, 0, 100000, 0),
        text: l.text.slice(0, 160),
      }))
      .slice(-MAX_LOG),
    offers: offers.slice(0, MAX_OFFERS),
    nextOffer: int(r.nextOffer, 1, 1e9, 1),
    over:
      r.over && typeof r.over === "object"
        ? {
            winner:
              typeof r.over.winner === "string" && ids.has(r.over.winner)
                ? r.over.winner
                : null,
            reason: String(r.over.reason ?? "").slice(0, 80),
          }
        : null,
  };
}

/** End the player's day: one strategist call per AI village still in the game (in parallel), then the
 *  deterministic turn. A failed or missing call keeps that village's current plan. */
export async function playTurn(
  spec: WorldSpec,
  b: Board,
  s: RivalsState,
  ask: ((fid: string, request: unknown) => Promise<unknown>) | null,
): Promise<{ state: RivalsState; calls: number }> {
  const ai = ask
    ? b.factions.filter((f) => !f.player && !s.factions[f.id]?.out)
    : [];
  const replies = await Promise.all(
    ai.map((f) =>
      (ask as NonNullable<typeof ask>)(
        f.id,
        strategistRequest(spec, b, s, f.id),
      ).catch(() => null),
    ),
  );
  const plans: Plans = {};
  ai.forEach((f, i) => {
    plans[f.id] = planOf(replies[i]);
  });
  return { state: endTurn(spec, b, s, plans), calls: ai.length };
}
