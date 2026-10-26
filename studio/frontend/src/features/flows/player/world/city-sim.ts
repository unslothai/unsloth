// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Mayor mode: a Micropolis-lite city simulation on the world tile map. Deterministic and pure; constants and
// rules are ported from Micropolis (GPL SimCity): s_sim.c SetValves / CollectTax, s_zone.c DoResidential /
// DoCommercial / DoIndustrial, s_traf.c MakeTraf, s_power.c DoPowerScan, s_scan.c PTLScan / CrimeScan / PopDenScan.
// Not ported: hospitals, police, fire, disasters, rail, ports, the 4x4 coal plant (ours is a 3x3 zone).

import type { WorldCity, WorldSpec } from "../../types";
import { SOLID, TILE_KINDS, type WorldMap } from "./procgen";

export const CITY_SAVE_VERSION = 1;
export type ZoneKind = "R" | "C" | "I" | "P";
export type CityTool = "road" | "wire" | ZoneKind | "bulldoze";
export const COSTS: Record<CityTool, number> = {
  road: 10,
  wire: 5,
  R: 100,
  C: 100,
  I: 100,
  P: 3000,
  bulldoze: 1,
};
export const TAX_TABLE = [
  200, 150, 120, 100, 80, 50, 30, 0, -10, -40, -100, -150, -200, -250, -300,
  -350, -400, -450, -500, -550, -600,
];
export const CENSUS_RATE = 4;
export const TAX_FREQ = 48;
export const CYCLES_PER_MONTH = 16;
export const CYCLE_MS = 1500;
export const MAX_DIS = 30;
export const POWER_PER_PLANT = 700;
export const R_VALVE_MAX = 2000;
export const CI_VALVE_MAX = 1500;
// Game level 0 (easy): ExtMarket 1.2, RLevels[0] 0.7, FLevels[0] 1.4.
const EXT_MARKET = 1.2;
const ROAD_LEVEL = 0.7;
const TAX_LEVEL = 1.4;
const PERIM_X = [-1, 0, 1, 2, 2, 2, 1, 0, -1, -2, -2, -2];
const PERIM_Y = [-2, -2, -2, -1, 0, 1, 2, 2, 2, 1, 0, -1];
const DX = [0, 1, 0, -1];
const DY = [-1, 0, 1, 0];

// Tile kinds in the city grid (saved as one character per tile).
export const EMPTY = 0;
export const ROAD = 1;
export const WIRE = 2;
export const ROADWIRE = 3;
export const ZONE = 4;
const GRID_CHARS = ".=-+z";

export const OVERLAYS = [
  "power",
  "traffic",
  "pollution",
  "land_value",
  "crime",
] as const;
export type Overlay = (typeof OVERLAYS)[number];
export const EVENTS = [
  "none",
  "festival",
  "fire",
  "strike",
  "flood",
  "boom",
] as const;
export type CityEvent = (typeof EVENTS)[number];

export interface CityZone {
  id: string;
  kind: ZoneKind;
  x: number; // centre tile
  y: number;
  pop: number; // R: 0-8 houses, then 16/24/32/40; C: 0-5; I: 0-4
  name?: string;
}

export interface Petition {
  month: number;
  citizen: string;
  zone: string;
  metric: string;
  value: number;
  text: string;
}

export interface CityState {
  v: number;
  w: number;
  h: number;
  grid: string;
  zones: CityZone[];
  next: number;
  funds: number;
  tax: number;
  cycle: number;
  month: number;
  rng: number;
  valves: { r: number; c: number; i: number };
  his: { res: [number, number]; com: [number, number]; ind: [number, number] };
  census: {
    res: number;
    com: number;
    ind: number;
    roads: number;
    total: number;
    lv: number;
    pollution: number;
    crime: number;
  };
  cash: { tax: number; roads: number; flow: number };
  cc: [number, number];
  maps: {
    traffic: string;
    pollution: string;
    land: string;
    crime: string;
    density: string;
    terrain: string;
    market: string;
  };
  trips: { ok: number; failed: number; noRoad: number };
  monthTrips: { ok: number; failed: number; noRoad: number };
  residents: Record<string, string>;
  advisor: { month: number; text: string; source: "model" | "local" } | null;
  petitions: Petition[];
  log: string[];
}

export function cityOf(spec: Pick<WorldSpec, "city">): WorldCity | null {
  const c = spec.city;
  return c && c.enabled !== false ? c : null;
}

function hex(a: Uint8Array): string {
  let s = "";
  for (let i = 0; i < a.length; i++) s += a[i].toString(16).padStart(2, "0");
  return s;
}

function unhex(s: unknown, into: Uint8Array): void {
  if (typeof s !== "string" || s.length !== into.length * 2) return;
  for (let i = 0; i < into.length; i++) {
    const v = Number.parseInt(s.slice(i * 2, i * 2 + 2), 16);
    into[i] = Number.isFinite(v) ? v : 0;
  }
}

const int = (v: unknown, d: number, lo = -1e9, hi = 1e9) =>
  typeof v === "number" && Number.isFinite(v)
    ? Math.max(lo, Math.min(hi, Math.trunc(v)))
    : d;

export function newCity(spec: WorldSpec): CityState {
  const c = cityOf(spec);
  const w = spec.width;
  const h = spec.height;
  const sim = new CitySim(
    {
      v: CITY_SAVE_VERSION,
      w,
      h,
      grid: ".".repeat(w * h),
      zones: [],
      next: 1,
      funds: c?.funds ?? 20000,
      tax: c?.tax ?? 7,
      cycle: 0,
      month: 0,
      rng: mixSeed(spec.seed),
      valves: { r: 0, c: 0, i: 0 },
      his: { res: [0, 0], com: [0, 0], ind: [0, 0] },
      census: {
        res: 0,
        com: 0,
        ind: 0,
        roads: 0,
        total: 0,
        lv: 0,
        pollution: 0,
        crime: 0,
      },
      cash: { tax: 0, roads: 0, flow: 0 },
      cc: [0, 0],
      maps: {
        traffic: "",
        pollution: "",
        land: "",
        crime: "",
        density: "",
        terrain: "",
        market: "",
      },
      trips: { ok: 0, failed: 0, noRoad: 0 },
      monthTrips: { ok: 0, failed: 0, noRoad: 0 },
      residents: {},
      advisor: null,
      petitions: [],
      log: [],
    },
    null,
  );
  sim.init();
  return sim.snapshot();
}

function trips(raw: unknown): CityState["trips"] {
  const t = (raw ?? {}) as Partial<CityState["trips"]>;
  return {
    ok: int(t.ok, 0, 0),
    failed: int(t.failed, 0, 0),
    noRoad: int(t.noRoad, 0, 0),
  };
}

/** Parse a saved city; null when it does not fit this map. */
export function parseCity(raw: unknown, spec: WorldSpec): CityState | null {
  const r = raw as Partial<CityState> | null;
  if (!r || typeof r !== "object" || r.v !== CITY_SAVE_VERSION) return null;
  const w = spec.width;
  const h = spec.height;
  if (r.w !== w || r.h !== h || typeof r.grid !== "string") return null;
  if (r.grid.length !== w * h || /[^.=+\-z]/.test(r.grid)) return null;
  const npcIds = new Set(spec.npcs.map((n) => n.id));
  const zones: CityZone[] = [];
  const ids = new Set<string>();
  for (const z of Array.isArray(r.zones) ? r.zones : []) {
    if (
      !z ||
      typeof z.id !== "string" ||
      ids.has(z.id) ||
      !["R", "C", "I", "P"].includes(z.kind) ||
      typeof z.x !== "number" ||
      typeof z.y !== "number" ||
      z.x < 1 ||
      z.y < 1 ||
      z.x > w - 2 ||
      z.y > h - 2
    )
      continue;
    ids.add(z.id);
    zones.push({
      id: z.id,
      kind: z.kind,
      x: z.x,
      y: z.y,
      pop: int(z.pop, 0, 0, 40),
      ...(typeof z.name === "string" ? { name: z.name.slice(0, 40) } : {}),
    });
  }
  const residents: Record<string, string> = {};
  for (const [id, zone] of Object.entries(r.residents ?? {}))
    if (npcIds.has(id) && ids.has(zone)) residents[id] = zone;
  const pair = (v: unknown): [number, number] =>
    Array.isArray(v) ? [int(v[0], 0), int(v[1], 0)] : [0, 0];
  const maps = (r.maps ?? {}) as Partial<CityState["maps"]>;
  const census = (r.census ?? {}) as Partial<CityState["census"]>;
  const s = (v: unknown) => (typeof v === "string" ? v : "");
  return {
    v: CITY_SAVE_VERSION,
    w,
    h,
    grid: r.grid,
    zones,
    next: int(r.next, zones.length + 1, 1),
    funds: int(r.funds, 0),
    tax: int(r.tax, 7, 0, 20),
    cycle: int(r.cycle, 0, 0),
    month: int(r.month, 0, 0),
    rng: int(r.rng, 1, 1, 0xffffffff),
    valves: {
      r: int(r.valves?.r, 0, -R_VALVE_MAX, R_VALVE_MAX),
      c: int(r.valves?.c, 0, -CI_VALVE_MAX, CI_VALVE_MAX),
      i: int(r.valves?.i, 0, -CI_VALVE_MAX, CI_VALVE_MAX),
    },
    his: {
      res: pair(r.his?.res),
      com: pair(r.his?.com),
      ind: pair(r.his?.ind),
    },
    census: {
      res: int(census.res, 0, 0),
      com: int(census.com, 0, 0),
      ind: int(census.ind, 0, 0),
      roads: int(census.roads, 0, 0),
      total: int(census.total, 0, 0),
      lv: int(census.lv, 0, 0),
      pollution: int(census.pollution, 0, 0),
      crime: int(census.crime, 0, 0),
    },
    cash: {
      tax: int(r.cash?.tax, 0),
      roads: int(r.cash?.roads, 0),
      flow: int(r.cash?.flow, 0),
    },
    cc: pair(r.cc),
    maps: {
      traffic: s(maps.traffic),
      pollution: s(maps.pollution),
      land: s(maps.land),
      crime: s(maps.crime),
      density: s(maps.density),
      terrain: s(maps.terrain),
      market: s(maps.market),
    },
    trips: trips(r.trips),
    monthTrips: trips(r.monthTrips),
    residents,
    advisor:
      r.advisor && typeof r.advisor.text === "string"
        ? {
            month: int(r.advisor.month, 0, 0),
            text: r.advisor.text.slice(0, 600),
            source: r.advisor.source === "model" ? "model" : "local",
          }
        : null,
    petitions: (Array.isArray(r.petitions) ? r.petitions : [])
      .filter(
        (p): p is Petition =>
          !!p &&
          typeof p.citizen === "string" &&
          typeof p.text === "string" &&
          typeof p.metric === "string",
      )
      .slice(-6),
    log: (Array.isArray(r.log) ? r.log : [])
      .filter((x): x is string => typeof x === "string")
      .slice(-8),
  };
}

/** Which tiles take buildings: walkable ground not blocked by scenery; trees and water add terrain value. */
export function cityGround(map: WorldMap): {
  buildable: Uint8Array;
  natural: Uint8Array;
} {
  const n = map.width * map.height;
  const buildable = new Uint8Array(n);
  const natural = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const kind = TILE_KINDS[map.tiles[i]];
    buildable[i] = SOLID.has(kind) || map.blocked?.[i] ? 0 : 1;
    natural[i] = kind === "tree" || kind === "water" ? 1 : 0;
  }
  return { buildable, natural };
}

export interface PlaceResult {
  ok: boolean;
  reason?: string;
}

export class CitySim {
  readonly w: number;
  readonly h: number;
  readonly hw: number;
  readonly hh: number;
  readonly qw: number;
  readonly qh: number;
  readonly sw: number;
  readonly sh: number;
  readonly kind: Uint8Array;
  readonly zoneAt: Int16Array;
  readonly powered: Uint8Array;
  readonly buildable: Uint8Array;
  readonly natural: Uint8Array;
  readonly traffic: Uint8Array;
  readonly pollution: Uint8Array;
  readonly land: Uint8Array;
  readonly crime: Uint8Array;
  readonly density: Uint8Array; // tem2 of PopDenScan; PopDensity = density << 1
  readonly terrain: Uint8Array;
  readonly market: Int16Array; // ComRate
  private readonly tem: Int16Array;
  private readonly tem2: Int16Array;
  private readonly qtem: Int16Array;
  private readonly stack: Int32Array;
  private readonly posX: Int16Array;
  private readonly posY: Int16Array;
  zones: CityZone[];
  private state: CityState;
  private seed: number;
  private posN = 0;
  private lDir = 5;
  private sx = 0;
  private sy = 0;
  powerUsed = 0;
  powerMax = 0;
  /** Last trip per zone index: 1 passed, 0 failed, -1 no road, 2 not tried. */
  trip: Int8Array;

  constructor(
    state: CityState,
    ground: { buildable: Uint8Array; natural: Uint8Array } | null,
  ) {
    const w = state.w;
    const h = state.h;
    this.w = w;
    this.h = h;
    this.hw = (w + 1) >> 1;
    this.hh = (h + 1) >> 1;
    this.qw = (w + 3) >> 2;
    this.qh = (h + 3) >> 2;
    this.sw = (w + 7) >> 3;
    this.sh = (h + 7) >> 3;
    const n = w * h;
    const hn = this.hw * this.hh;
    this.kind = new Uint8Array(n);
    this.zoneAt = new Int16Array(n);
    this.powered = new Uint8Array(n);
    this.buildable = ground?.buildable ?? new Uint8Array(n).fill(1);
    this.natural = ground?.natural ?? new Uint8Array(n);
    this.traffic = new Uint8Array(hn);
    this.pollution = new Uint8Array(hn);
    this.land = new Uint8Array(hn);
    this.crime = new Uint8Array(hn);
    this.density = new Uint8Array(hn);
    this.terrain = new Uint8Array(this.qw * this.qh);
    this.market = new Int16Array(this.sw * this.sh);
    this.tem = new Int16Array(hn);
    this.tem2 = new Int16Array(hn);
    this.qtem = new Int16Array(this.qw * this.qh);
    this.stack = new Int32Array(n);
    this.posX = new Int16Array(MAX_DIS + 1);
    this.posY = new Int16Array(MAX_DIS + 1);
    this.state = { ...structuredClone(state), zones: [] };
    this.zones = state.zones.map((z) => ({ ...z }));
    this.trip = new Int8Array(Math.max(64, this.zones.length * 2)).fill(2);
    this.seed = state.rng || 1;
    for (let i = 0; i < n; i++) {
      const k = GRID_CHARS.indexOf(state.grid[i] ?? ".");
      this.kind[i] = k < 0 ? EMPTY : k;
    }
    unhex(state.maps.traffic, this.traffic);
    unhex(state.maps.pollution, this.pollution);
    unhex(state.maps.land, this.land);
    unhex(state.maps.crime, this.crime);
    unhex(state.maps.density, this.density);
    unhex(state.maps.terrain, this.terrain);
    const market = new Uint8Array(this.market.length);
    unhex(state.maps.market, market);
    if (state.maps.market)
      for (let i = 0; i < market.length; i++) this.market[i] = market[i] - 128;
    this.reindex();
    this.powerScan();
  }

  get funds(): number {
    return this.state.funds;
  }
  get tax(): number {
    return this.state.tax;
  }
  get month(): number {
    return this.state.month;
  }
  get cycle(): number {
    return this.state.cycle;
  }
  get valves(): CityState["valves"] {
    return this.state.valves;
  }
  get census(): CityState["census"] {
    return this.state.census;
  }
  get cash(): CityState["cash"] {
    return this.state.cash;
  }
  get trips(): CityState["trips"] {
    return this.state.trips;
  }
  get monthTrips(): CityState["monthTrips"] {
    return this.state.monthTrips;
  }
  get his(): CityState["his"] {
    return this.state.his;
  }
  get residents(): Record<string, string> {
    return this.state.residents;
  }

  /** Census and overlays for a brand new city (DoSimInit without a map scan: nothing is zoned yet). */
  init(): void {
    this.setValves();
    this.clearCensus();
    this.powerScan();
    this.ptlScan();
    this.crimeScan();
    this.popDenScan();
  }

  rand16(): number {
    let x = this.seed;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.seed = x >>> 0 || 1;
    return (this.seed >>> 8) & 0xffff;
  }

  rand16Signed(): number {
    const v = this.rand16();
    return v > 32767 ? v - 65536 : v;
  }

  /** Micropolis Rand(range): 0..range inclusive. */
  rand(range: number): number {
    return this.rand16() % (range + 1);
  }

  private reindex(): void {
    this.zoneAt.fill(0);
    const w = this.w;
    this.zones.forEach((z, i) => {
      for (let y = z.y - 1; y <= z.y + 1; y++)
        for (let x = z.x - 1; x <= z.x + 1; x++) this.zoneAt[y * w + x] = i + 1;
    });
    if (this.trip.length < this.zones.length) {
      const t = new Int8Array(this.zones.length * 2).fill(2);
      t.set(this.trip);
      this.trip = t;
    }
  }

  zoneOf(x: number, y: number): CityZone | null {
    if (x < 0 || y < 0 || x >= this.w || y >= this.h) return null;
    const i = this.zoneAt[y * this.w + x];
    return i ? this.zones[i - 1] : null;
  }

  zoneById(id: string): CityZone | null {
    return this.zones.find((z) => z.id === id) ?? null;
  }

  isPowered(z: CityZone): boolean {
    return this.powered[z.y * this.w + z.x] === 1;
  }

  /** Player tools. Zones are 3x3 around the clicked tile, as in Micropolis. */
  place(tool: CityTool, x: number, y: number): PlaceResult {
    const w = this.w;
    if (x < 0 || y < 0 || x >= w || y >= this.h)
      return { ok: false, reason: "Off the map" };
    const cost = COSTS[tool];
    if (this.state.funds < cost)
      return { ok: false, reason: "Not enough funds" };
    const i = y * w + x;
    const k = this.kind[i];
    if (tool === "bulldoze") {
      if (k === EMPTY) return { ok: false, reason: "Nothing to clear" };
      if (k === ZONE) {
        const zi = this.zoneAt[i] - 1;
        const z = this.zones[zi];
        for (let yy = z.y - 1; yy <= z.y + 1; yy++)
          for (let xx = z.x - 1; xx <= z.x + 1; xx++)
            this.kind[yy * w + xx] = EMPTY;
        this.zones.splice(zi, 1);
        this.trip.copyWithin(zi, zi + 1);
        for (const [npc, zone] of Object.entries(this.state.residents))
          if (zone === z.id) delete this.state.residents[npc];
        this.reindex();
      } else this.kind[i] = EMPTY;
      this.state.funds -= cost;
      this.powerScan();
      return { ok: true };
    }
    if (tool === "road" || tool === "wire") {
      const other = tool === "road" ? WIRE : ROAD;
      const same = tool === "road" ? ROAD : WIRE;
      if (k === same || k === ROADWIRE)
        return { ok: false, reason: "Already built" };
      if (k === ZONE) return { ok: false, reason: "A zone is there" };
      if (!this.buildable[i])
        return { ok: false, reason: "Cannot build there" };
      this.kind[i] = k === other ? ROADWIRE : same;
      this.state.funds -= cost;
      this.powerScan();
      return { ok: true };
    }
    if (x < 1 || y < 1 || x > w - 2 || y > this.h - 2)
      return { ok: false, reason: "Too close to the edge" };
    for (let yy = y - 1; yy <= y + 1; yy++)
      for (let xx = x - 1; xx <= x + 1; xx++) {
        const j = yy * w + xx;
        if (this.kind[j] !== EMPTY || !this.buildable[j])
          return { ok: false, reason: "A zone needs 3x3 clear ground" };
      }
    for (let yy = y - 1; yy <= y + 1; yy++)
      for (let xx = x - 1; xx <= x + 1; xx++) this.kind[yy * w + xx] = ZONE;
    const id = `${tool}${this.state.next++}`;
    this.zones.push({ id, kind: tool, x, y, pop: 0 });
    this.reindex();
    this.trip[this.zones.length - 1] = 2;
    this.state.funds -= cost;
    this.powerScan();
    return { ok: true };
  }

  setTax(tax: number): void {
    this.state.tax = Math.max(0, Math.min(20, Math.round(tax)));
  }

  isRoad(x: number, y: number): boolean {
    if (x < 0 || y < 0 || x >= this.w || y >= this.h) return false;
    const k = this.kind[y * this.w + x];
    return k === ROAD || k === ROADWIRE;
  }

  /** SetValves (s_sim.c), game level 0. */
  setValves(): void {
    const s = this.state;
    const c = s.census;
    const resHis1 = s.his.res[1];
    const comHis1 = s.his.com[1];
    const indHis1 = s.his.ind[1];
    const normResPop = c.res / 8;
    c.total = Math.trunc(normResPop + c.com + c.ind);
    const employment = normResPop ? (comHis1 + indHis1) / normResPop : 1;
    const migration = normResPop * (employment - 1);
    const births = normResPop * 0.02;
    const pjResPop = normResPop + migration + births;
    let laborBase = comHis1 + indHis1 ? resHis1 / (comHis1 + indHis1) : 1;
    if (laborBase > 1.3) laborBase = 1.3;
    if (laborBase < 0) laborBase = 0;
    const intMarket = (normResPop + c.com + c.ind) / 3.7;
    const pjComPop = intMarket * laborBase;
    let pjIndPop = c.ind * laborBase * EXT_MARKET;
    if (pjIndPop < 5) pjIndPop = 5;
    let rRatio = normResPop ? pjResPop / normResPop : 1.3;
    let cRatio = c.com ? pjComPop / c.com : pjComPop;
    let iRatio = c.ind ? pjIndPop / c.ind : pjIndPop;
    if (rRatio > 2) rRatio = 2;
    if (cRatio > 2) cRatio = 2;
    if (iRatio > 2) iRatio = 2;
    const t = TAX_TABLE[Math.min(20, s.tax)];
    rRatio = (rRatio - 1) * 600 + t;
    cRatio = (cRatio - 1) * 600 + t;
    iRatio = (iRatio - 1) * 600 + t;
    const v = s.valves;
    const nudge = (valve: number, ratio: number, max: number) => {
      let out = valve;
      if (ratio > 0 && out < max) out = Math.trunc(out + ratio);
      if (ratio < 0 && out > -max) out = Math.trunc(out + ratio);
      return Math.max(-max, Math.min(max, out));
    };
    v.r = nudge(v.r, rRatio, R_VALVE_MAX);
    v.c = nudge(v.c, cRatio, CI_VALVE_MAX);
    v.i = nudge(v.i, iRatio, CI_VALVE_MAX);
  }

  private clearCensus(): void {
    const c = this.state.census;
    c.res = 0;
    c.com = 0;
    c.ind = 0;
    c.roads = 0;
  }

  /** One Micropolis cycle (the 16 phases of Simulate, every scan at speed 1). True when a month ended. */
  step(): boolean {
    const s = this.state;
    s.cycle++;
    if (!(s.cycle & 1)) this.setValves();
    this.clearCensus();
    this.mapScan();
    if (s.cycle % CENSUS_RATE === 0) this.takeCensus();
    let monthEnded = false;
    if (s.cycle % CYCLES_PER_MONTH === 0) {
      this.collectTax();
      s.month++;
      s.monthTrips = s.trips;
      s.trips = { ok: 0, failed: 0, noRoad: 0 };
      monthEnded = true;
    }
    this.decTrafficMem();
    this.powerScan();
    this.ptlScan();
    this.crimeScan();
    this.popDenScan();
    return monthEnded;
  }

  private mapScan(): void {
    const c = this.state.census;
    for (let i = 0; i < this.kind.length; i++)
      if (this.kind[i] === ROAD || this.kind[i] === ROADWIRE) c.roads++;
    for (let zi = 0; zi < this.zones.length; zi++) {
      const z = this.zones[zi];
      const powered = this.isPowered(z);
      if (z.kind === "R") this.doResidential(zi, z, powered);
      else if (z.kind === "C") this.doCommercial(zi, z, powered);
      else if (z.kind === "I") this.doIndustrial(zi, z, powered);
    }
  }

  private cell(z: CityZone): number {
    return (z.y >> 1) * this.hw + (z.x >> 1);
  }

  private traf(zi: number, z: CityZone, source: 0 | 1 | 2): number {
    const t = this.makeTraf(z, source);
    this.trip[zi] = t;
    const trips = this.state.trips;
    if (t === 1) trips.ok++;
    else if (t === 0) trips.failed++;
    else trips.noRoad++;
    return t;
  }

  private doResidential(zi: number, z: CityZone, powered: boolean): void {
    const tpop = z.pop;
    this.state.census.res += tpop;
    const trf = tpop > this.rand(35) ? this.traf(zi, z, 0) : 1;
    if (trf === -1) {
      this.resOut(z);
      return;
    }
    if (z.pop <= 8 || !(this.rand16() & 7)) {
      let zscore = this.state.valves.r + this.evalRes(z, trf);
      if (!powered) zscore = -500;
      if (zscore > -350 && i16(zscore - 26380) > this.rand16Signed()) {
        this.resIn(z);
        return;
      }
      if (zscore < 350 && i16(zscore + 26380) < this.rand16Signed())
        this.resOut(z);
    }
  }

  private evalRes(z: CityZone, trf: number): number {
    if (trf < 0) return -3000;
    const c = this.cell(z);
    let value = this.land[c] - this.pollution[c];
    value = value < 0 ? 0 : value << 5;
    if (value > 6000) value = 6000;
    return value - 3000;
  }

  private resIn(z: CityZone): void {
    const c = this.cell(z);
    if (this.pollution[c] > 128) return;
    if (z.pop <= 8) {
      if (z.pop < 8) z.pop++;
      else if (this.density[c] << 1 > 64) z.pop = 16;
      return;
    }
    if (z.pop < 40) z.pop += 8;
  }

  resOut(z: CityZone): void {
    const pop = z.pop;
    if (!pop) return;
    if (pop > 16) z.pop = pop - 8;
    else if (pop === 16) z.pop = 8;
    else z.pop = pop - 1;
  }

  private doCommercial(zi: number, z: CityZone, powered: boolean): void {
    const tpop = z.pop;
    this.state.census.com += tpop;
    const trf = tpop > this.rand(5) ? this.traf(zi, z, 1) : 1;
    if (trf === -1) {
      this.ciOut(z);
      return;
    }
    if (!(this.rand16() & 7)) {
      const loc =
        trf < 0 ? -3000 : this.market[(z.y >> 3) * this.sw + (z.x >> 3)];
      let zscore = this.state.valves.c + loc;
      if (!powered) zscore = -500;
      if (trf && zscore > -350 && i16(zscore - 26380) > this.rand16Signed()) {
        if (z.pop <= this.land[this.cell(z)] >> 5 && z.pop < 5) z.pop++;
        return;
      }
      if (zscore < 350 && i16(zscore + 26380) < this.rand16Signed())
        this.ciOut(z);
    }
  }

  ciOut(z: CityZone): void {
    if (z.pop > 0) z.pop--;
  }

  private doIndustrial(zi: number, z: CityZone, powered: boolean): void {
    const tpop = z.pop;
    this.state.census.ind += tpop;
    const trf = tpop > this.rand(5) ? this.traf(zi, z, 2) : 1;
    if (trf === -1) {
      this.rand16();
      this.ciOut(z);
      return;
    }
    if (!(this.rand16() & 7)) {
      let zscore = this.state.valves.i + (trf < 0 ? -1000 : 0);
      if (!powered) zscore = -500;
      if (zscore > -350 && i16(zscore - 26380) > this.rand16Signed()) {
        this.rand16();
        if (z.pop < 4) z.pop++;
        return;
      }
      if (zscore < 350 && i16(zscore + 26380) < this.rand16Signed()) {
        this.rand16();
        this.ciOut(z);
      }
    }
  }

  /** MakeTraf: 1 trip passed, 0 failed, -1 no road on the zone's perimeter. */
  makeTraf(z: CityZone, source: 0 | 1 | 2): number {
    this.posN = 0;
    let found = false;
    for (let k = 0; k < 12; k++) {
      const tx = z.x + PERIM_X[k];
      const ty = z.y + PERIM_Y[k];
      if (this.isRoad(tx, ty)) {
        this.sx = tx;
        this.sy = ty;
        found = true;
        break;
      }
    }
    if (!found) return -1;
    if (this.tryDrive(source)) {
      this.setTrafMem();
      return 1;
    }
    return 0;
  }

  private tryDrive(source: number): boolean {
    this.lDir = 5;
    for (let z = 0; z < MAX_DIS; z++) {
      if (this.tryGo(z)) {
        if (this.driveDone(source)) return true;
      } else if (this.posN) {
        this.posN--;
        z += 3;
      } else return false;
    }
    return false;
  }

  private tryGo(z: number): boolean {
    const rdir = this.rand16() & 3;
    for (let x = rdir; x < rdir + 4; x++) {
      const d = x & 3;
      if (d === this.lDir) continue;
      if (this.isRoad(this.sx + DX[d], this.sy + DY[d])) {
        this.sx += DX[d];
        this.sy += DY[d];
        this.lDir = (d + 2) & 3;
        if (z & 1 && this.posN < MAX_DIS) {
          this.posN++;
          this.posX[this.posN] = this.sx;
          this.posY[this.posN] = this.sy;
        }
        return true;
      }
    }
    return false;
  }

  /** R goes to C/I/plants, C to developed R/C/I, I to developed R (DriveDone's TARGL/TARGH). */
  private driveDone(source: number): boolean {
    for (let d = 0; d < 4; d++) {
      const z = this.zoneOf(this.sx + DX[d], this.sy + DY[d]);
      if (!z) continue;
      if (source === 0 && z.kind !== "R") return true;
      if (source === 1 && z.kind !== "P" && (z.kind !== "R" || z.pop > 0))
        return true;
      if (source === 2 && z.kind === "R" && z.pop > 0) return true;
    }
    return false;
  }

  private setTrafMem(): void {
    for (let k = this.posN; k > 0; k--) {
      const x = this.posX[k];
      const y = this.posY[k];
      if (this.isRoad(x, y)) {
        const c = (y >> 1) * this.hw + (x >> 1);
        this.traffic[c] = Math.min(240, this.traffic[c] + 50);
      }
    }
    this.posN = 0;
  }

  private decTrafficMem(): void {
    const t = this.traffic;
    for (let i = 0; i < t.length; i++) {
      const z = t[i];
      if (!z) continue;
      t[i] = z > 24 ? (z > 200 ? z - 34 : z - 24) : 0;
    }
  }

  private takeCensus(): void {
    const s = this.state;
    s.his.res[1] = s.his.res[0];
    s.his.com[1] = s.his.com[0];
    s.his.ind[1] = s.his.ind[0];
    s.his.res[0] = Math.trunc(s.census.res / 8);
    s.his.com[0] = s.census.com;
    s.his.ind[0] = s.census.ind;
  }

  /** CollectTax (yearly in Micropolis), paid each month as CYCLES_PER_MONTH / TAX_FREQ of the year. */
  collectTax(): void {
    const s = this.state;
    const b = budget(s.census.total, s.census.lv, s.tax, s.census.roads);
    s.cash = b;
    s.funds += b.flow;
  }

  /** DoPowerScan: flood fill from the plants through wires and zones, 700 tiles per plant. */
  powerScan(): void {
    const w = this.w;
    const h = this.h;
    this.powered.fill(0);
    let plants = 0;
    let top = 0;
    for (const z of this.zones)
      if (z.kind === "P") {
        plants++;
        this.stack[top++] = z.y * w + z.x;
      }
    this.powerMax = plants * POWER_PER_PLANT;
    let used = 0;
    for (let k = 0; k < top; k++) this.powered[this.stack[k]] = 1;
    used = top;
    while (top > 0) {
      const i = this.stack[--top];
      const x = i % w;
      const y = (i - x) / w;
      for (let d = 0; d < 4; d++) {
        const nx = x + DX[d];
        const ny = y + DY[d];
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const j = ny * w + nx;
        const k = this.kind[j];
        if (this.powered[j] || (k !== WIRE && k !== ROADWIRE && k !== ZONE))
          continue;
        if (used >= this.powerMax) {
          this.powerUsed = used;
          return;
        }
        this.powered[j] = 1;
        used++;
        this.stack[top++] = j;
      }
    }
    this.powerUsed = used;
  }

  private pValue(i: number, x: number, y: number): number {
    const k = this.kind[i];
    if (k === ROAD || k === ROADWIRE) {
      const d = this.traffic[(y >> 1) * this.hw + (x >> 1)];
      return d >= 192 ? 75 : d >= 64 ? 50 : 0;
    }
    if (k === ZONE) {
      const z = this.zones[this.zoneAt[i] - 1];
      if (z.kind === "P") return 100;
      if (z.kind === "I" && z.pop > 0) return 50;
    }
    return 0;
  }

  private disCC(x: number, y: number): number {
    const [cx, cy] = this.state.cc;
    const z = Math.abs(x - cx) + Math.abs(y - cy);
    return z > 32 ? 32 : z;
  }

  /** Micropolis DoSmooth / DoSmooth2: (four neighbours + centre) >> 2, capped at 255. */
  private smooth(src: Int16Array, dst: Int16Array): void {
    const hw = this.hw;
    const hh = this.hh;
    for (let y = 0; y < hh; y++)
      for (let x = 0; x < hw; x++) {
        const i = y * hw + x;
        let z = 0;
        if (x > 0) z += src[i - 1];
        if (x < hw - 1) z += src[i + 1];
        if (y > 0) z += src[i - hw];
        if (y < hh - 1) z += src[i + hw];
        z = (z + src[i]) >> 2;
        dst[i] = z > 255 ? 255 : z;
      }
  }

  /** PTLScan: pollution, terrain and land value at half resolution. */
  private ptlScan(): void {
    const w = this.w;
    const hw = this.hw;
    const qtem = this.qtem;
    qtem.fill(0);
    let lvTot = 0;
    let lvNum = 0;
    for (let y = 0; y < this.hh; y++)
      for (let x = 0; x < hw; x++) {
        let plevel = 0;
        let lvFlag = 0;
        for (let my = y << 1; my <= (y << 1) + 1 && my < this.h; my++)
          for (let mx = x << 1; mx <= (x << 1) + 1 && mx < w; mx++) {
            const i = my * w + mx;
            if (this.kind[i] === EMPTY) {
              if (this.natural[i]) qtem[(y >> 1) * this.qw + (x >> 1)] += 15;
              continue;
            }
            plevel += this.pValue(i, mx, my);
            lvFlag++;
          }
        const c = y * hw + x;
        this.tem[c] = plevel > 255 ? 255 : plevel;
        if (lvFlag) {
          let dis = (34 - this.disCC(x, y)) << 2;
          dis += this.terrain[(y >> 1) * this.qw + (x >> 1)];
          dis -= this.pollution[c];
          if (this.crime[c] > 190) dis -= 20;
          if (dis > 250) dis = 250;
          if (dis < 1) dis = 1;
          this.land[c] = dis;
          lvTot += dis;
          lvNum++;
        } else this.land[c] = 0;
      }
    this.state.census.lv = lvNum ? Math.trunc(lvTot / lvNum) : 0;
    this.smooth(this.tem, this.tem2);
    this.smooth(this.tem2, this.tem);
    let pTot = 0;
    let pNum = 0;
    for (let c = 0; c < this.pollution.length; c++) {
      const z = this.tem[c];
      this.pollution[c] = z;
      if (z) {
        pNum++;
        pTot += z;
      }
    }
    this.state.census.pollution = pNum ? Math.trunc(pTot / pNum) : 0;
    const qw = this.qw;
    const qh = this.qh;
    for (let y = 0; y < qh; y++)
      for (let x = 0; x < qw; x++) {
        const i = y * qw + x;
        let z = 0;
        if (x > 0) z += qtem[i - 1];
        if (x < qw - 1) z += qtem[i + 1];
        if (y > 0) z += qtem[i - qw];
        if (y < qh - 1) z += qtem[i + qw];
        this.terrain[i] = Math.min(255, ((z >> 2) + qtem[i]) >> 1);
      }
  }

  /** CrimeScan without police: 128 - land value + population density, clamped to 0..250. */
  private crimeScan(): void {
    let tot = 0;
    let num = 0;
    for (let c = 0; c < this.crime.length; c++) {
      const lv = this.land[c];
      if (!lv) {
        this.crime[c] = 0;
        continue;
      }
      let z = 128 - lv + (this.density[c] << 1);
      if (z > 300) z = 300;
      if (z > 250) z = 250;
      if (z < 0) z = 0;
      this.crime[c] = z;
      tot += z;
      num++;
    }
    this.state.census.crime = num ? Math.trunc(tot / num) : 0;
  }

  /** PopDenScan, DistIntMarket and the city centre of mass. */
  private popDenScan(): void {
    this.tem.fill(0);
    let xt = 0;
    let yt = 0;
    let zt = 0;
    for (const z of this.zones) {
      const pden = z.kind === "R" ? z.pop : z.kind === "P" ? 0 : z.pop << 3;
      this.tem[this.cell(z)] = Math.min(254, pden << 3);
      xt += z.x;
      yt += z.y;
      zt++;
    }
    this.smooth(this.tem, this.tem2);
    this.smooth(this.tem2, this.tem);
    this.smooth(this.tem, this.tem2);
    for (let c = 0; c < this.density.length; c++)
      this.density[c] = this.tem2[c];
    for (let y = 0; y < this.sh; y++)
      for (let x = 0; x < this.sw; x++)
        this.market[y * this.sw + x] = 64 - (this.disCC(x << 2, y << 2) << 2);
    const ccx = zt ? Math.trunc(xt / zt) : this.hw;
    const ccy = zt ? Math.trunc(yt / zt) : this.hh;
    this.state.cc = [ccx >> 1, ccy >> 1];
  }

  /** Fixed event menu; deterministic effects, no randomness. */
  applyEvent(kind: CityEvent, zoneId: string, size: number): string | null {
    const n = Math.max(1, Math.min(3, Math.round(size)));
    const z = zoneId ? this.zoneById(zoneId) : null;
    const v = this.state.valves;
    const clamp = (x: number, m: number) => Math.max(-m, Math.min(m, x));
    let text: string | null = null;
    if (kind === "festival") {
      v.r = clamp(v.r + 100 * n, R_VALVE_MAX);
      v.c = clamp(v.c + 100 * n, CI_VALVE_MAX);
      text = `A festival lifts the mood (+${100 * n} residential and commercial demand)`;
    } else if (kind === "boom") {
      v.i = clamp(v.i + 200 * n, CI_VALVE_MAX);
      v.c = clamp(v.c + 100 * n, CI_VALVE_MAX);
      text = `A trade boom (+${200 * n} industrial demand)`;
    } else if ((kind === "fire" || kind === "flood") && z && z.kind !== "P") {
      for (let k = 0; k < n; k++)
        if (z.kind === "R") this.resOut(z);
        else this.ciOut(z);
      text = `A ${kind} hits ${z.name ?? z.id}`;
    } else if (kind === "strike" && z && (z.kind === "I" || z.kind === "C")) {
      this.ciOut(z);
      v.i = clamp(v.i - 100 * n, CI_VALVE_MAX);
      text = `Workers strike at ${z.name ?? z.id} (-${100 * n} industrial demand)`;
    }
    if (text) this.state.log = [...this.state.log, text].slice(-8);
    return text;
  }

  /** The numbers at a tile for overlays and petitions. */
  at(
    x: number,
    y: number,
  ): {
    land_value: number;
    pollution: number;
    crime: number;
    traffic: number;
  } {
    const c = (y >> 1) * this.hw + (x >> 1);
    return {
      land_value: this.land[c],
      pollution: this.pollution[c],
      crime: this.crime[c],
      traffic: this.traffic[c],
    };
  }

  hasRoad(z: CityZone): boolean {
    for (let k = 0; k < 12; k++)
      if (this.isRoad(z.x + PERIM_X[k], z.y + PERIM_Y[k])) return true;
    return false;
  }

  /** Busiest road cell next to a zone. */
  zoneTraffic(z: CityZone): number {
    let best = 0;
    for (let k = 0; k < 12; k++) {
      const x = z.x + PERIM_X[k];
      const y = z.y + PERIM_Y[k];
      if (this.isRoad(x, y))
        best = Math.max(best, this.traffic[(y >> 1) * this.hw + (x >> 1)]);
    }
    return best;
  }

  /** Give homeless villagers a house in a grown R zone, one per house, in zone order. */
  moveIn(npcIds: string[]): string[] {
    const res = this.state.residents;
    const count = new Map<string, number>();
    for (const zone of Object.values(res))
      count.set(zone, (count.get(zone) ?? 0) + 1);
    const moved: string[] = [];
    for (const id of npcIds) {
      if (res[id]) continue;
      const home = this.zones.find(
        (z) => z.kind === "R" && z.pop > (count.get(z.id) ?? 0),
      );
      if (!home) break;
      res[id] = home.id;
      count.set(home.id, (count.get(home.id) ?? 0) + 1);
      moved.push(id);
    }
    return moved;
  }

  nameZone(id: string, name: string): boolean {
    const z = this.zoneById(id);
    if (!z || z.name) return false;
    z.name = name.slice(0, 40);
    return true;
  }

  setAdvisor(text: string, source: "model" | "local"): void {
    this.state.advisor = { month: this.state.month, text, source };
  }

  addPetitions(ps: Petition[]): void {
    this.state.petitions = [...this.state.petitions, ...ps].slice(-6);
  }

  get advisor(): CityState["advisor"] {
    return this.state.advisor;
  }
  get petitions(): Petition[] {
    return this.state.petitions;
  }
  get log(): string[] {
    return this.state.log;
  }

  snapshot(): CityState {
    const s = this.state;
    let grid = "";
    for (let i = 0; i < this.kind.length; i++) grid += GRID_CHARS[this.kind[i]];
    const market = new Uint8Array(this.market.length);
    for (let i = 0; i < market.length; i++) market[i] = this.market[i] + 128;
    return {
      ...s,
      v: CITY_SAVE_VERSION,
      w: this.w,
      h: this.h,
      grid,
      zones: this.zones.map((z) => ({ ...z })),
      rng: this.seed,
      valves: { ...s.valves },
      his: {
        res: [...s.his.res] as [number, number],
        com: [...s.his.com] as [number, number],
        ind: [...s.his.ind] as [number, number],
      },
      census: { ...s.census },
      cash: { ...s.cash },
      cc: [...s.cc] as [number, number],
      maps: {
        traffic: hex(this.traffic),
        pollution: hex(this.pollution),
        land: hex(this.land),
        crime: hex(this.crime),
        density: hex(this.density),
        terrain: hex(this.terrain),
        market: hex(market),
      },
      trips: { ...s.trips },
      monthTrips: { ...s.monthTrips },
      residents: { ...s.residents },
      petitions: [...s.petitions],
      log: [...s.log],
    };
  }
}

/** Spread a small world seed over 32 bits (murmur3 finaliser) so xorshift starts well mixed. */
function mixSeed(seed: number): number {
  let h = (seed ^ 0x9e3779b9) >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  h = (h ^ (h >>> 16)) >>> 0;
  return h % 0x7fffffff || 1;
}

function i16(v: number): number {
  const x = v & 0xffff;
  return x > 32767 ? x - 65536 : x;
}

/** Monthly budget: CollectTax's yearly TaxFund and RoadFund (game level 0), scaled to one month. */
export function budget(
  totalPop: number,
  lvAverage: number,
  tax: number,
  roads: number,
): { tax: number; roads: number; flow: number } {
  if (!totalPop) return { tax: 0, roads: 0, flow: 0 };
  const taxYear = Math.trunc(
    Math.trunc((totalPop * lvAverage) / 120) * tax * TAX_LEVEL,
  );
  const roadYear = Math.trunc(roads * ROAD_LEVEL);
  const t = Math.trunc((taxYear * CYCLES_PER_MONTH) / TAX_FREQ);
  const r = Math.trunc((roadYear * CYCLES_PER_MONTH) / TAX_FREQ);
  return { tax: t, roads: r, flow: t - r };
}
