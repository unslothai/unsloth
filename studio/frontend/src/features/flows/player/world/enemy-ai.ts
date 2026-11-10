// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type Combat,
  ENEMY_DAMAGE,
  ENEMY_HEALTH,
  ENEMY_RANGE,
} from "./action3d";
import type { Noise } from "./action-verbs";

export type AiState =
  | "idle"
  | "patrol"
  | "alert"
  | "chase"
  | "attack"
  | "pain"
  | "flee"
  | "return";
export type Tactic = "hold" | "flank" | "cover";
export type BarkKind = "see" | "hear" | "lose" | "pain" | "flee" | "attack";

export const THINK_S = 0.1;
export const VIEW_RANGE = 12;
export const VIEW_COS = Math.cos(1.0);
export const CROUCH_SEEN = 0.5;
export const FEEL_RANGE = 1.2;
export const REACT_S = 0.35;
export const PAIN_S = 0.35;
export const LOSE_S = 4;
export const IDLE_S = 2.5;
export const LEASH = 14;
export const FLEE_HP = 0.3;
export const COVER_HP = 0.6;
export const MELEE_REACH = 1.3;
export const MELEE_HIT = 10;
export const MELEE_COOLDOWN_S = 1;
export const RANGED_COOLDOWN_S = 2;
export const BARK_GAP_S = 3;
const SPEED: Record<AiState, number> = {
  idle: 0,
  patrol: 1.2,
  alert: 0,
  chase: 2.4,
  attack: 1.8,
  pain: 0,
  flee: 2.7,
  return: 1.6,
};
const PATH_MAX = 48;
const PATROL_R = 4;
const FLANK_R = 6;
const COVER_R = 8;
const MAX_COVER = 1024;
const EYE = 1.1;

export const BARKS: Record<BarkKind, readonly string[]> = {
  see: ["There you are!", "Found you!", "Intruder!", "Over here!"],
  hear: [
    "What was that?",
    "Who's there?",
    "I heard something.",
    "Show yourself.",
  ],
  lose: [
    "Where did they go?",
    "Lost them.",
    "Must have been nothing.",
    "Back to my post.",
  ],
  pain: ["Argh!", "Ow!", "You'll pay for that!"],
  flee: ["I'm out of here!", "Fall back!", "Not today!"],
  attack: ["Take this!", "Got you now!", "Stand still!"],
};

/** A raid's bark list ({event, text}, events spot / lost / hurt / flank / retreat) mapped onto our kinds. */
const RAID_EVENTS: Record<string, BarkKind> = {
  spot: "see",
  lost: "lose",
  hurt: "pain",
  flank: "attack",
  retreat: "flee",
};

/** The world's bark lines over the built-in ones, per kind (or a raid's tagged list); bad entries are ignored. */
export function barkPool(raw: unknown): Record<BarkKind, readonly string[]> {
  const out = { ...BARKS };
  if (Array.isArray(raw)) {
    const by: Partial<Record<BarkKind, string[]>> = {};
    for (const b of raw) {
      const kind = RAID_EVENTS[(b as { event?: string })?.event ?? ""];
      const text = (b as { text?: unknown })?.text;
      if (kind && typeof text === "string" && text.trim())
        (by[kind] ??= []).push(text);
    }
    raw = by;
  }
  if (!raw || typeof raw !== "object") return out;
  for (const k of Object.keys(BARKS) as BarkKind[]) {
    const v = (raw as Record<string, unknown>)[k];
    const lines = Array.isArray(v)
      ? v
          .filter((s): s is string => typeof s === "string" && s.trim() !== "")
          .map((s) => s.trim().slice(0, 80))
          .slice(0, 12)
      : [];
    if (lines.length) out[k] = lines;
  }
  return out;
}

export interface Grid {
  width: number;
  height: number;
  blocked: Uint8Array;
}

export const gridOf = (
  width: number,
  height: number,
  solid: (x: number, y: number) => boolean,
): Grid => {
  const blocked = new Uint8Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      blocked[y * width + x] = solid(x, y) ? 1 : 0;
  return { width, height, blocked };
};

/** Grid A* (4-neighbour, Manhattan) over reusable typed arrays: no allocation per search. */
export class Pathfinder {
  private g: Float32Array;
  private parent: Int32Array;
  private seen: Uint32Array;
  private closed: Uint32Array;
  private heap: Int32Array;
  private prio: Float32Array;
  private n = 0;
  private gen = 0;
  expanded = 0;
  readonly size: number;

  constructor(size: number) {
    this.size = size;
    this.g = new Float32Array(size);
    this.parent = new Int32Array(size);
    this.seen = new Uint32Array(size);
    this.closed = new Uint32Array(size);
    this.heap = new Int32Array(size * 4 + 4);
    this.prio = new Float32Array(size * 4 + 4);
  }

  private push(i: number, f: number): void {
    if (this.n >= this.heap.length) return;
    let k = this.n++;
    while (k > 0) {
      const up = (k - 1) >> 1;
      if (this.prio[up] <= f) break;
      this.heap[k] = this.heap[up];
      this.prio[k] = this.prio[up];
      k = up;
    }
    this.heap[k] = i;
    this.prio[k] = f;
  }

  private pop(): number {
    const top = this.heap[0];
    const n = --this.n;
    const li = this.heap[n];
    const lf = this.prio[n];
    let k = 0;
    for (;;) {
      let c = 2 * k + 1;
      if (c >= n) break;
      if (c + 1 < n && this.prio[c + 1] < this.prio[c]) c++;
      if (this.prio[c] >= lf) break;
      this.heap[k] = this.heap[c];
      this.prio[k] = this.prio[c];
      k = c;
    }
    this.heap[k] = li;
    this.prio[k] = lf;
    return top;
  }

  /**
   * Steps from (sx, sy) to (tx, ty) as tile indices into `out` (start excluded, goal last; only the first
   * `out.length` when longer). Returns the full path length, 0 when already there, -1 when unreachable.
   */
  find(
    grid: Grid,
    sx: number,
    sy: number,
    tx: number,
    ty: number,
    out: Int32Array,
    maxExpand = 6000,
  ): number {
    const w = grid.width;
    const h = grid.height;
    if (sx < 0 || sy < 0 || sx >= w || sy >= h) return -1;
    if (tx < 0 || ty < 0 || tx >= w || ty >= h) return -1;
    const start = sy * w + sx;
    const goal = ty * w + tx;
    if (start === goal) return 0;
    if (grid.blocked[goal]) return -1;
    if (++this.gen === 0xffffffff) {
      this.seen.fill(0);
      this.closed.fill(0);
      this.gen = 1;
    }
    const gen = this.gen;
    this.n = 0;
    this.g[start] = 0;
    this.seen[start] = gen;
    this.parent[start] = -1;
    this.push(start, Math.abs(sx - tx) + Math.abs(sy - ty));
    this.expanded = 0;
    while (this.n > 0) {
      const cur = this.pop();
      if (this.closed[cur] === gen) continue;
      this.closed[cur] = gen;
      if (cur === goal) break;
      if (++this.expanded > maxExpand) return -1;
      const cx = cur % w;
      const cy = (cur / w) | 0;
      const gc = this.g[cur] + 1;
      for (let d = 0; d < 4; d++) {
        const nx = cx + (d === 0 ? 1 : d === 1 ? -1 : 0);
        const ny = cy + (d === 2 ? 1 : d === 3 ? -1 : 0);
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const ni = ny * w + nx;
        if (grid.blocked[ni] || this.closed[ni] === gen) continue;
        if (this.seen[ni] === gen && this.g[ni] <= gc) continue;
        this.seen[ni] = gen;
        this.g[ni] = gc;
        this.parent[ni] = cur;
        const hh = Math.abs(nx - tx) + Math.abs(ny - ty);
        // Ties go to the node nearer the goal, so equal-cost fronts do not all expand.
        this.push(ni, gc + hh + hh * 1e-3);
      }
    }
    if (this.closed[goal] !== gen) return -1;
    const len = this.g[goal];
    let k = len - 1;
    for (let i = goal; i !== start; i = this.parent[i], k--)
      if (k < out.length) out[k] = i;
    return len;
  }
}

export function clearLine(
  grid: Grid,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
): boolean {
  const dx = Math.abs(x1 - x0);
  const dy = -Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let err = dx + dy;
  let x = x0;
  let y = y0;
  for (;;) {
    if ((x !== x0 || y !== y0) && (x !== x1 || y !== y1))
      if (grid.blocked[y * grid.width + x]) return false;
    if (x === x1 && y === y1) return true;
    const e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y += sy;
    }
  }
}

export interface Agent {
  id: string;
  x: number;
  y: number;
  homeX: number;
  homeY: number;
  fx: number;
  fy: number;
  state: AiState;
  since: number;
  think: number;
  tactic: Tactic;
  lastX: number;
  lastY: number;
  lastAt: number;
  seen: boolean;
  heard: boolean;
  heardSeq: number;
  health: number;
  ready: number;
  guard: boolean;
  down: boolean;
  moving: boolean;
  path: Int32Array;
  pathLen: number;
  pathAt: number;
  goal: number;
  rng: number;
  bark: string;
  barkSeq: number;
  barkAt: number;
  attacks: number;
}

export interface PlayerSense {
  x: number;
  y: number;
  z: number;
  h: number;
  fx: number;
  fy: number;
  crouched: boolean;
  hidden: boolean;
}

/** Line of sight in the action world: the distance to the first wall or prop along a ray (wallDistance). */
export type WallFn = (
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
  max: number,
) => number;

export interface Brain {
  grid: Grid;
  finder: Pathfinder;
  agents: Agent[];
  byId: Map<string, Agent>;
  clock: number;
  /** Newest noise seq stepped so far: a hostile added later starts past it. */
  noiseSeq: number;
  cover: Int32Array;
  coverLen: number;
  influence: Float32Array;
  barks: Record<BarkKind, readonly string[]>;
  wall: WallFn;
  ground: (x: number, y: number) => number;
  /** Safe places: a hostile standing in one never attacks. */
  safe: (x: number, y: number) => boolean;
}

export function createBrain(
  grid: Grid,
  wall: WallFn,
  ground: (x: number, y: number) => number,
  cover: (x: number, y: number) => boolean,
  barks?: unknown,
): Brain {
  const b: Brain = {
    grid,
    finder: new Pathfinder(grid.width * grid.height),
    agents: [],
    byId: new Map(),
    clock: 0,
    noiseSeq: 0,
    cover: new Int32Array(MAX_COVER),
    coverLen: 0,
    influence: new Float32Array(grid.width * grid.height),
    barks: barkPool(barks),
    wall,
    ground,
    safe: () => false,
  };
  setGrid(b, grid, cover);
  return b;
}

/** New walls (a door opened or closed, props edited): swap the grid and rebuild cover points next to `cover` tiles. */
export function setGrid(
  b: Brain,
  grid: Grid,
  cover: (x: number, y: number) => boolean,
): void {
  b.grid = grid;
  if (grid.width * grid.height !== b.finder.size) {
    b.finder = new Pathfinder(grid.width * grid.height);
    b.influence = new Float32Array(grid.width * grid.height);
  }
  b.coverLen = 0;
  const w = grid.width;
  for (let y = 0; y < grid.height && b.coverLen < MAX_COVER; y++)
    for (let x = 0; x < w && b.coverLen < MAX_COVER; x++) {
      if (grid.blocked[y * w + x]) continue;
      if (
        (x + 1 < w && cover(x + 1, y)) ||
        (x > 0 && cover(x - 1, y)) ||
        (y + 1 < grid.height && cover(x, y + 1)) ||
        (y > 0 && cover(x, y - 1))
      )
        b.cover[b.coverLen++] = y * w + x;
    }
  for (const a of b.agents) {
    a.pathLen = 0;
    a.goal = -1;
  }
}

const hash = (s: string) => {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++)
    h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0 || 1;
};

export function addAgent(
  b: Brain,
  id: string,
  x: number,
  y: number,
  guard = false,
): Agent {
  const known = b.byId.get(id);
  if (known) return known;
  const a: Agent = {
    id,
    x,
    y,
    homeX: x,
    homeY: y,
    fx: 0,
    fy: 1,
    state: "idle",
    since: b.clock,
    // Staggered so ten hostiles do not all think in the same frame.
    think: b.clock + (b.agents.length % 10) * (THINK_S / 10),
    tactic: "hold",
    lastX: x,
    lastY: y,
    lastAt: Number.NEGATIVE_INFINITY,
    seen: false,
    heard: false,
    heardSeq: b.noiseSeq,
    health: ENEMY_HEALTH,
    ready: 0,
    guard,
    down: false,
    moving: false,
    path: new Int32Array(PATH_MAX),
    pathLen: 0,
    pathAt: 0,
    goal: -1,
    rng: hash(id),
    bark: "",
    barkSeq: 0,
    barkAt: Number.NEGATIVE_INFINITY,
    attacks: 0,
  };
  b.agents.push(a);
  b.byId.set(id, a);
  return a;
}

const rand = (a: Agent) => {
  a.rng ^= a.rng << 13;
  a.rng >>>= 0;
  a.rng ^= a.rng >>> 17;
  a.rng ^= a.rng << 5;
  a.rng >>>= 0;
  return a.rng / 4294967296;
};

function bark(b: Brain, a: Agent, kind: BarkKind, force = false): void {
  if (!force && b.clock - a.barkAt < BARK_GAP_S) return;
  const pool = b.barks[kind];
  if (!pool.length) return;
  a.bark = pool[(hash(a.id) + a.barkSeq) % pool.length];
  a.barkSeq++;
  a.barkAt = b.clock;
}

function enter(b: Brain, a: Agent, s: AiState): void {
  if (a.state === s) return;
  a.state = s;
  a.since = b.clock;
  a.goal = -1;
  a.pathLen = 0;
  a.pathAt = 0;
}

const tileOf = (b: Brain, x: number, y: number) =>
  Math.floor(y) * b.grid.width + Math.floor(x);

function go(b: Brain, a: Agent, tile: number): boolean {
  if (tile < 0) return false;
  if (a.goal === tile && a.pathAt < a.pathLen) return true;
  const w = b.grid.width;
  const len = b.finder.find(
    b.grid,
    Math.floor(a.x),
    Math.floor(a.y),
    tile % w,
    (tile / w) | 0,
    a.path,
  );
  a.goal = tile;
  a.pathAt = 0;
  a.pathLen = len > 0 ? Math.min(len, PATH_MAX) : 0;
  return len >= 0;
}

const arrived = (b: Brain, a: Agent) =>
  a.goal < 0 || (a.pathAt >= a.pathLen && tileOf(b, a.x, a.y) === a.goal);

export function canSee(b: Brain, a: Agent, p: PlayerSense): boolean {
  if (p.hidden) return false;
  const dx = p.x - a.x;
  const dy = p.y - a.y;
  const d = Math.hypot(dx, dy);
  // Up close it feels the player whichever way it faces, but not through a wall or a closed door.
  if (d >= FEEL_RANGE) {
    const range = VIEW_RANGE * (p.crouched ? CROUCH_SEEN : 1);
    if (d > range) return false;
    // Right after contact it tracks the player all around; a hostile that lost them looks where it faces.
    const aware =
      (a.state === "chase" || a.state === "attack") && b.clock - a.lastAt < 1;
    if (!aware && (dx * a.fx + dy * a.fy) / d < VIEW_COS) return false;
  }
  const eye = b.ground(a.x, a.y) + EYE;
  const mid = p.z + 0.7 * p.h;
  return b.wall(a.x, a.y, eye, dx, dy, mid - eye, d) >= d;
}

/** The loudest new noise this agent can hear (radius fades with distance); true when it heard one. */
export function hear(a: Agent, noises: readonly Noise[]): boolean {
  let best = 0;
  let top = a.heardSeq;
  for (const n of noises) {
    if (n.seq <= a.heardSeq) continue;
    if (n.seq > top) top = n.seq;
    const d = Math.hypot(n.x - a.x, n.y - a.y);
    const level = 1 - d / n.radius;
    if (level > best) {
      best = level;
      a.lastX = n.x;
      a.lastY = n.y;
    }
  }
  a.heardSeq = top;
  return best > 0;
}

/** Tactical scores near the player: in front of their gaze and next to other engaged hostiles is worse. */
function weigh(b: Brain, a: Agent, p: PlayerSense): void {
  const w = b.grid.width;
  const px = Math.floor(p.x);
  const py = Math.floor(p.y);
  for (let y = py - FLANK_R; y <= py + FLANK_R; y++)
    for (let x = px - FLANK_R; x <= px + FLANK_R; x++) {
      if (x < 0 || y < 0 || x >= w || y >= b.grid.height) continue;
      const i = y * w + x;
      const dx = x + 0.5 - p.x;
      const dy = y + 0.5 - p.y;
      const d = Math.hypot(dx, dy) || 1;
      let v = Math.max(0, (dx * p.fx + dy * p.fy) / d);
      for (const o of b.agents)
        if (o !== a && !o.down && (o.state === "chase" || o.state === "attack"))
          v +=
            1.5 * Math.max(0, 1 - Math.hypot(x + 0.5 - o.x, y + 0.5 - o.y) / 3);
      b.influence[i] = v;
    }
}

/** A flanking tile: open, 2.5 to 6 tiles from the player with a clear shot, away from their gaze and from allies. */
export function pickFlank(b: Brain, a: Agent, p: PlayerSense): number {
  weigh(b, a, p);
  const w = b.grid.width;
  const px = Math.floor(p.x);
  const py = Math.floor(p.y);
  let best = -1;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (let y = py - FLANK_R; y <= py + FLANK_R; y++)
    for (let x = px - FLANK_R; x <= px + FLANK_R; x++) {
      if (x < 0 || y < 0 || x >= w || y >= b.grid.height) continue;
      const i = y * w + x;
      if (b.grid.blocked[i]) continue;
      const d = Math.hypot(x + 0.5 - p.x, y + 0.5 - p.y);
      if (d < 2.5 || d > FLANK_R) continue;
      if (!clearLine(b.grid, px, py, x, y)) continue;
      const score =
        -b.influence[i] -
        0.1 * Math.abs(d - 4) -
        0.05 * Math.hypot(x + 0.5 - a.x, y + 0.5 - a.y);
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
  return best;
}

export function pickCover(b: Brain, a: Agent, p: PlayerSense): number {
  const w = b.grid.width;
  const px = Math.floor(p.x);
  const py = Math.floor(p.y);
  let best = -1;
  let bestD = COVER_R;
  for (let k = 0; k < b.coverLen; k++) {
    const i = b.cover[k];
    const x = i % w;
    const y = (i / w) | 0;
    const d = Math.hypot(x + 0.5 - a.x, y + 0.5 - a.y);
    if (d >= bestD || clearLine(b.grid, px, py, x, y)) continue;
    best = i;
    bestD = d;
  }
  return best;
}

function fleeTile(b: Brain, a: Agent, p: PlayerSense): number {
  const w = b.grid.width;
  let best = -1;
  let bestD = Math.hypot(a.x - p.x, a.y - p.y);
  for (let k = 0; k < 24; k++) {
    const ang = (k / 24) * Math.PI * 2;
    const x = Math.floor(a.x + Math.cos(ang) * 6);
    const y = Math.floor(a.y + Math.sin(ang) * 6);
    if (x < 0 || y < 0 || x >= w || y >= b.grid.height) continue;
    if (b.grid.blocked[y * w + x]) continue;
    const d = Math.hypot(x + 0.5 - p.x, y + 0.5 - p.y);
    if (d > bestD) {
      best = y * w + x;
      bestD = d;
    }
  }
  return best;
}

/** Utility choice once the player is in sight: attack, flee when hurt badly, or take cover when hurt and seen. */
export function choose(b: Brain, a: Agent, p: PlayerSense): AiState {
  const hp = a.health / ENEMY_HEALTH;
  const d = Math.hypot(p.x - a.x, p.y - a.y);
  const flee = hp < FLEE_HP ? 1 + (FLEE_HP - hp) : 0;
  const reach = d <= MELEE_REACH || d <= ENEMY_RANGE;
  const attack = (reach ? 0.9 : 0.4) * (0.4 + 0.6 * hp);
  const cover =
    hp < COVER_HP && b.coverLen > 0 && a.tactic !== "cover"
      ? 0.5 + (COVER_HP - hp)
      : 0;
  const chase = reach ? 0 : 0.6;
  if (flee >= attack && flee >= cover && flee >= chase) return "flee";
  if (cover > attack && cover >= chase) {
    a.tactic = "cover";
    return "attack";
  }
  if (chase > attack) return "chase";
  return "attack";
}

function think(
  b: Brain,
  a: Agent,
  p: PlayerSense,
  c: Combat,
  noises: readonly Noise[],
): void {
  const e = c.enemies[a.id];
  const health = e ? e.health : ENEMY_HEALTH;
  const hurt = health < a.health;
  a.health = health;
  a.seen = canSee(b, a, p);
  a.heard = hear(a, noises);
  if (a.heard) a.lastAt = b.clock;
  if (a.seen) {
    a.lastX = p.x;
    a.lastY = p.y;
    a.lastAt = b.clock;
  }
  const t = b.clock - a.since;
  const lost = b.clock - a.lastAt;
  if (hurt && a.state !== "flee") {
    a.lastX = p.x;
    a.lastY = p.y;
    a.lastAt = b.clock;
    enter(b, a, "pain");
    bark(b, a, "pain", true);
    return;
  }
  const leash = Math.hypot(a.x - a.homeX, a.y - a.homeY) > LEASH;
  switch (a.state) {
    case "idle":
    case "patrol":
    case "return":
      if (a.seen || a.heard) {
        if (a.health / ENEMY_HEALTH < FLEE_HP && a.seen) {
          enter(b, a, "flee");
          bark(b, a, "flee");
        } else {
          enter(b, a, "alert");
          bark(b, a, a.seen ? "see" : "hear");
        }
      } else if (a.state === "return") {
        if (!go(b, a, tileOf(b, a.homeX, a.homeY)) || arrived(b, a))
          enter(b, a, "idle");
      } else if (a.state === "idle" && !a.guard && t > IDLE_S) {
        enter(b, a, "patrol");
        const ang = rand(a) * Math.PI * 2;
        const r = 1 + rand(a) * (PATROL_R - 1);
        const tx = Math.floor(a.homeX + Math.cos(ang) * r);
        const ty = Math.floor(a.homeY + Math.sin(ang) * r);
        const w = b.grid.width;
        if (
          tx < 0 ||
          ty < 0 ||
          tx >= w ||
          ty >= b.grid.height ||
          !go(b, a, ty * w + tx)
        )
          enter(b, a, "idle");
      } else if (a.state === "patrol" && arrived(b, a)) enter(b, a, "idle");
      return;
    case "alert":
      if (t < REACT_S) return;
      a.tactic = "hold";
      if (a.seen) enter(b, a, choose(b, a, p));
      else enter(b, a, "chase");
      return;
    case "pain":
      if (t < PAIN_S) return;
      a.tactic = "hold";
      if (a.health / ENEMY_HEALTH < FLEE_HP) {
        enter(b, a, "flee");
        bark(b, a, "flee", true);
      } else enter(b, a, "chase");
      return;
    case "flee":
      if (a.goal < 0 || arrived(b, a)) {
        const tile = fleeTile(b, a, p);
        if (tile < 0 || !go(b, a, tile)) a.goal = -1;
      }
      if (lost > LOSE_S) {
        enter(b, a, "return");
        bark(b, a, "lose");
      }
      return;
    case "chase":
      if (
        leash ||
        (!a.seen && (lost > LOSE_S || (lost > 1.5 && arrived(b, a))))
      ) {
        enter(b, a, "return");
        bark(b, a, "lose");
        return;
      }
      if (a.seen) {
        const next = choose(b, a, p);
        if (next !== "chase") {
          enter(b, a, next);
          return;
        }
      }
      go(b, a, tileOf(b, a.lastX, a.lastY));
      return;
    case "attack": {
      if (!a.seen) {
        a.tactic = "hold";
        enter(b, a, "chase");
        return;
      }
      const next = choose(b, a, p);
      if (next !== "attack") {
        enter(b, a, next);
        if (next === "flee") bark(b, a, "flee", true);
        return;
      }
      const d = Math.hypot(p.x - a.x, p.y - a.y);
      if (a.tactic === "cover") {
        const tile = a.goal >= 0 ? a.goal : pickCover(b, a, p);
        if (tile < 0 || !go(b, a, tile)) a.tactic = "hold";
      } else if (d > MELEE_REACH) {
        let engaged = false;
        for (const o of b.agents)
          if (
            o !== a &&
            !o.down &&
            (o.state === "attack" || o.state === "chase")
          )
            engaged = true;
        if (engaged && a.tactic !== "flank") {
          a.tactic = "flank";
          go(b, a, pickFlank(b, a, p));
        } else if (!engaged && d > ENEMY_RANGE * 0.8)
          go(b, a, tileOf(b, p.x, p.y));
      }
      if (b.clock >= a.ready && !p.hidden && !b.safe(a.x, a.y)) {
        const melee = d <= MELEE_REACH;
        if (melee || d <= ENEMY_RANGE) {
          a.ready = b.clock + (melee ? MELEE_COOLDOWN_S : RANGED_COOLDOWN_S);
          c.health = Math.max(0, c.health - (melee ? MELEE_HIT : ENEMY_DAMAGE));
          a.attacks++;
          bark(b, a, "attack");
        }
      }
      return;
    }
  }
}

function move(b: Brain, a: Agent, p: PlayerSense, dt: number): void {
  a.moving = false;
  const speed = SPEED[a.state];
  if (a.state === "alert" || (a.state === "attack" && a.seen)) {
    const dx = (a.state === "alert" ? a.lastX : p.x) - a.x;
    const dy = (a.state === "alert" ? a.lastY : p.y) - a.y;
    const d = Math.hypot(dx, dy);
    if (d > 1e-6) {
      a.fx = dx / d;
      a.fy = dy / d;
    }
  }
  if (!speed || a.pathAt >= a.pathLen) return;
  let left = speed * dt;
  const w = b.grid.width;
  while (left > 0 && a.pathAt < a.pathLen) {
    const i = a.path[a.pathAt];
    const tx = (i % w) + 0.5;
    const ty = ((i / w) | 0) + 0.5;
    const dx = tx - a.x;
    const dy = ty - a.y;
    const d = Math.hypot(dx, dy);
    if (d <= left) {
      a.x = tx;
      a.y = ty;
      left -= d;
      a.pathAt++;
    } else {
      a.x += (dx / d) * left;
      a.y += (dy / d) * left;
      left = 0;
    }
    if (d > 1e-6 && !(a.state === "attack" && a.seen)) {
      a.fx = dx / d;
      a.fy = dy / d;
    }
    a.moving = true;
  }
}

export const engaged = (a: Agent) => a.state !== "idle" && a.state !== "patrol";

/**
 * Who moves this NPC: when the hostile is calm, village life does and the agent just follows it (its post
 * becomes where it stands); returns true while combat owns it.
 */
export function follow(a: Agent, x: number, y: number): boolean {
  if (engaged(a) || a.down) return true;
  a.x = a.homeX = x;
  a.y = a.homeY = y;
  a.pathLen = 0;
  return false;
}

/** Decide at 10 Hz, move every frame. `noises` is the player's noise ring (shots, throws, doors, steps). */
export function brainStep(
  b: Brain,
  dt: number,
  p: PlayerSense,
  c: Combat,
  noises: readonly Noise[],
): void {
  b.clock += dt;
  for (const n of noises) if (n.seq > b.noiseSeq) b.noiseSeq = n.seq;
  for (const a of b.agents) {
    const down = (c.enemies[a.id]?.down ?? 0) > c.clock;
    if (down) {
      // Knocked out, it hears nothing: on waking, older noises are already behind it.
      for (const n of noises) if (n.seq > a.heardSeq) a.heardSeq = n.seq;
      if (!a.down) {
        a.down = true;
        a.x = a.homeX;
        a.y = a.homeY;
        a.health = ENEMY_HEALTH;
        a.lastAt = Number.NEGATIVE_INFINITY;
        a.tactic = "hold";
        enter(b, a, "idle");
      }
      continue;
    }
    a.down = false;
    if (b.clock >= a.think) {
      a.think += THINK_S;
      if (a.think < b.clock) a.think = b.clock + THINK_S;
      think(b, a, p, c, noises);
    }
    move(b, a, p, dt);
  }
}
