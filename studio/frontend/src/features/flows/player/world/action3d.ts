// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import type { WorldProp3d, WorldPropKind } from "../../types";

export const STEP_S = 1 / 60;
const MAX_STEPS = 6;
export const RADIUS = 0.24;
export const STAND_H = 1.45;
export const CROUCH_H = 0.85;
export const STEP_UP = 0.4;
const GRAVITY = 16;
const JUMP_V = 5.2;
const COYOTE_S = 0.1;
const JUMP_BUFFER_S = 0.12;
const WALL_H = 6;
const MAX_SLOPE = 1.2;

export interface Box {
  cx: number;
  cy: number;
  hx: number;
  hy: number;
  cos: number;
  sin: number;
  bottom: number;
  top: number;
  id?: string;
}

const PROP_BOX: Record<WorldPropKind, [number, number, number]> = {
  house: [0.9, 0.9, 1.35],
  tree: [0.25, 0.25, 1.7],
  lamp: [0.1, 0.1, 1.7],
  fence: [0.96, 0.1, 0.55],
  well: [0.76, 0.76, 1.28],
  crate: [0.52, 0.52, 0.51],
  rock: [0.7, 0.7, 0.54],
  bench: [0.8, 0.3, 0.6],
  stall: [1, 0.75, 1.1],
  door: [0.96, 0.14, 1.9],
  chest: [0.7, 0.46, 0.5],
  lever: [0.2, 0.2, 0.9],
  model: [0.9, 0.9, 1],
};

export interface ActionWorld {
  width: number;
  height: number;
  ground(x: number, y: number): number;
  solidTile(x: number, y: number): boolean;
  cells: Map<number, Box[]>;
  boxes: Box[];
}

export function propBox(
  p: WorldProp3d,
  ground: (x: number, y: number) => number,
): Box | null {
  if (p.solid === false) return null;
  const [w, d, h] = PROP_BOX[p.kind] ?? PROP_BOX.model;
  const s = p.scale ?? 1;
  const a = ((p.rot ?? 0) * Math.PI) / 180;
  const bottom = ground(p.x, p.y);
  return {
    cx: p.x,
    cy: p.y,
    hx: (w * s) / 2,
    hy: (d * s) / 2,
    cos: Math.cos(a),
    sin: Math.sin(a),
    bottom,
    top: bottom + h * s,
    id: p.id,
  };
}

export function buildActionWorld(
  width: number,
  height: number,
  ground: (x: number, y: number) => number,
  solidTile: (x: number, y: number) => boolean,
  props: readonly WorldProp3d[],
): ActionWorld {
  const boxes = props
    .map((p) => propBox(p, ground))
    .filter((b): b is Box => b !== null);
  const cells = new Map<number, Box[]>();
  for (const b of boxes) {
    const r = Math.hypot(b.hx, b.hy);
    for (let y = Math.floor(b.cy - r); y <= Math.floor(b.cy + r); y++)
      for (let x = Math.floor(b.cx - r); x <= Math.floor(b.cx + r); x++) {
        const k = y * width + x;
        const list = cells.get(k);
        if (list) list.push(b);
        else cells.set(k, [b]);
      }
  }
  return { width, height, ground, solidTile, cells, boxes };
}

const NONE: Box[] = [];
const near: Box[] = [];

// Returns a shared array (or a cell's own list): read it before the next call, never keep or mutate it.
function nearBoxes(w: ActionWorld, x: number, y: number, r: number): Box[] {
  const x0 = Math.floor(x - r);
  const x1 = Math.floor(x + r);
  const y0 = Math.floor(y - r);
  const y1 = Math.floor(y + r);
  if (x0 === x1 && y0 === y1) return w.cells.get(y0 * w.width + x0) ?? NONE;
  near.length = 0;
  for (let cy = y0; cy <= y1; cy++)
    for (let cx = x0; cx <= x1; cx++) {
      const list = w.cells.get(cy * w.width + cx);
      if (list) for (const b of list) if (!near.includes(b)) near.push(b);
    }
  return near;
}

// The capsule can reach into a door's box from the next tile, which a tile check misses.
export function capsuleOverlapsProp(
  p: WorldProp3d,
  ground: (x: number, y: number) => number,
  x: number,
  y: number,
): boolean {
  const b = propBox(p, ground);
  return b !== null && circleHitsBox(b, x, y, RADIUS);
}

export function circleHitsBox(
  b: Box,
  x: number,
  y: number,
  r: number,
): boolean {
  const dx = x - b.cx;
  const dy = y - b.cy;
  // three.js turns +x toward -z (tile -y) for a positive Y rotation, as props-scene3d renders the prop.
  const lx = dx * b.cos - dy * b.sin;
  const ly = dx * b.sin + dy * b.cos;
  const px = Math.max(-b.hx, Math.min(b.hx, lx));
  const py = Math.max(-b.hy, Math.min(b.hy, ly));
  return (lx - px) ** 2 + (ly - py) ** 2 < r * r;
}

function tileBlocked(w: ActionWorld, x: number, y: number, r: number): boolean {
  const x0 = Math.floor(x - r);
  const x1 = Math.floor(x + r);
  const y0 = Math.floor(y - r);
  const y1 = Math.floor(y + r);
  return (
    w.solidTile(x0, y0) ||
    w.solidTile(x1, y0) ||
    w.solidTile(x0, y1) ||
    w.solidTile(x1, y1)
  );
}

function floorAt(w: ActionWorld, x: number, y: number, reach: number): number {
  let floor = w.ground(x, y);
  for (const b of nearBoxes(w, x, y, RADIUS))
    if (b.top <= reach && b.top > floor && circleHitsBox(b, x, y, RADIUS))
      floor = b.top;
  return floor;
}

function blocked(
  w: ActionWorld,
  x: number,
  y: number,
  z: number,
  h: number,
): boolean {
  if (tileBlocked(w, x, y, RADIUS)) return z < w.ground(x, y) + WALL_H;
  for (const b of nearBoxes(w, x, y, RADIUS))
    if (
      b.top > z + STEP_UP &&
      b.bottom < z + h &&
      circleHitsBox(b, x, y, RADIUS)
    )
      return true;
  return false;
}

export interface Body {
  x: number;
  y: number;
  z: number;
  vz: number;
  grounded: boolean;
  crouched: boolean;
  coyote: number;
  jumpBuffer: number;
  acc: number;
  jumpQueued?: boolean;
}

export interface MoveInput {
  dx: number;
  dy: number;
  sprint: boolean;
  crouch: boolean;
  jump: boolean;
  limit?: number;
}

export const newBody = (w: ActionWorld, x: number, y: number): Body => ({
  x,
  y,
  z: floorAt(w, x, y, Number.POSITIVE_INFINITY),
  vz: 0,
  grounded: true,
  crouched: false,
  coyote: 0,
  jumpBuffer: 0,
  acc: 0,
});

export const bodyHeight = (b: Body) => (b.crouched ? CROUCH_H : STAND_H);

const SUB_MOVE = 0.2;

function step(w: ActionWorld, b: Body, input: MoveInput, walk: number): number {
  const dt = STEP_S;
  if (input.crouch) b.crouched = true;
  else if (b.crouched && !blocked(w, b.x, b.y, b.z, STAND_H))
    b.crouched = false;
  const h = bodyHeight(b);
  const speed = walk * (b.crouched ? 0.5 : input.sprint ? 1.75 : 1);
  const len = Math.hypot(input.dx, input.dy);
  const go = Math.min(speed * dt, input.limit ?? Number.POSITIVE_INFINITY);
  const mx = len > 0 ? (input.dx / len) * go : 0;
  const my = len > 0 ? (input.dy / len) * go : 0;
  // Sub-moves no longer than SUB_MOVE, so a fast sprint cannot pass through a thin prop between two steps.
  const subs = Math.max(
    1,
    Math.ceil(Math.max(Math.abs(mx), Math.abs(my)) / SUB_MOVE),
  );
  for (let i = 0; i < subs; i++)
    for (let axis = 0; axis < 2; axis++) {
      const ax = axis === 0 ? mx / subs : 0;
      const ay = axis === 0 ? 0 : my / subs;
      if (!ax && !ay) continue;
      const nx = b.x + ax;
      const ny = b.y + ay;
      if (!blocked(w, nx, ny, b.z, h)) {
        const floor = floorAt(w, nx, ny, b.z + STEP_UP);
        const rise = floor - b.z;
        if (
          b.grounded &&
          rise > STEP_UP &&
          rise > MAX_SLOPE * Math.hypot(ax, ay)
        )
          continue;
        b.x = nx;
        b.y = ny;
        if (rise > 0 && rise <= STEP_UP) b.z = floor;
        else if (b.grounded && rise < 0 && rise > -STEP_UP) b.z = floor;
      }
    }
  b.jumpBuffer = input.jump ? JUMP_BUFFER_S : Math.max(0, b.jumpBuffer - dt);
  if (b.jumpBuffer > 0 && (b.grounded || b.coyote > 0) && !b.crouched) {
    b.vz = JUMP_V;
    b.grounded = false;
    b.coyote = 0;
    b.jumpBuffer = 0;
  }
  b.vz -= GRAVITY * dt;
  let z = b.z + b.vz * dt;
  if (b.vz > 0)
    for (const box of nearBoxes(w, b.x, b.y, RADIUS))
      if (
        box.bottom >= b.z + h - 1e-6 &&
        box.bottom < z + h &&
        circleHitsBox(box, b.x, b.y, RADIUS)
      ) {
        z = box.bottom - h;
        b.vz = 0;
      }
  const floor = floorAt(w, b.x, b.y, Math.max(z, b.z) + 1e-6);
  if (z <= floor) {
    b.z = floor;
    b.vz = 0;
    if (!b.grounded) b.coyote = 0;
    b.grounded = true;
  } else {
    if (b.grounded) b.coyote = COYOTE_S;
    else b.coyote = Math.max(0, b.coyote - dt);
    b.grounded = false;
    b.z = z;
  }
  return len > 0 ? go : 0;
}

export function advance(
  w: ActionWorld,
  b: Body,
  input: MoveInput,
  dt: number,
  walk: number,
): number {
  b.acc = Math.min(b.acc + dt, MAX_STEPS * STEP_S);
  let n = 0;
  let jump = input.jump || Boolean(b.jumpQueued);
  b.jumpQueued = false;
  let limit = input.limit;
  while (b.acc >= STEP_S - 1e-9) {
    const moved = step(w, b, { ...input, jump, limit }, walk);
    if (limit !== undefined) limit = Math.max(0, limit - moved);
    jump = false;
    b.acc -= STEP_S;
    n++;
  }
  if (jump) b.jumpQueued = true;
  return n;
}


export const MAX_HEALTH = 100;
export const START_AMMO = 24;
export const MAX_AMMO = 60;
export const SHOT_COOLDOWN_S = 0.3;
export const SHOT_RANGE = 24;
export const SHOT_DAMAGE = 34;
export const ENEMY_HEALTH = 100;
export const ENEMY_DAMAGE = 6;
export const ENEMY_COOLDOWN_S = 2;
export const ENEMY_RANGE = 6;
export const KNOCKOUT_S = 45;
export const PICKUP_HEALTH = 35;
export const PICKUP_AMMO = 12;
const BODY_MID = 0.6;
const AIM_ASSIST_RAD = 0.35;

export interface Combat {
  health: number;
  ammo: number;
  cooldown: number;
  enemies: Record<string, { health: number; down: number; cooldown: number }>;
  taken: string[];
  clock: number;
}

export const newCombat = (): Combat => ({
  health: MAX_HEALTH,
  ammo: START_AMMO,
  cooldown: 0,
  enemies: {},
  taken: [],
  clock: 0,
});

export interface Target {
  id: string;
  x: number;
  y: number;
  z: number;
  hostile: boolean;
}

export function wallDistance(
  w: ActionWorld,
  ox: number,
  oy: number,
  oz: number,
  dx: number,
  dy: number,
  dz: number,
  max: number,
): number {
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const uz = dz / len;
  const step = 0.05;
  for (let t = step; t <= max; t += step) {
    const x = ox + ux * t;
    const y = oy + uy * t;
    const z = oz + uz * t;
    const tx = Math.floor(x);
    const ty = Math.floor(y);
    if (w.solidTile(tx, ty) || z < w.ground(x, y)) return t;
    const cell = w.cells.get(ty * w.width + tx);
    if (cell)
      for (const b of cell)
        if (z >= b.bottom && z <= b.top && circleHitsBox(b, x, y, 1e-3))
          return t;
  }
  return Number.POSITIVE_INFINITY;
}

export function shoot(
  w: ActionWorld,
  c: Combat,
  eye: { x: number; y: number; z: number },
  dir: { x: number; y: number; z: number },
  targets: readonly Target[],
): { fired: boolean; hit: string | null; downed: boolean } {
  if (c.cooldown > 0 || c.ammo <= 0)
    return { fired: false, hit: null, downed: false };
  c.ammo -= 1;
  c.cooldown = SHOT_COOLDOWN_S;
  const flat = Math.hypot(dir.x, dir.y) || 1e-9;
  let best: Target | null = null;
  let bestT = Number.POSITIVE_INFINITY;
  for (const t of targets) {
    if ((c.enemies[t.id]?.down ?? 0) > c.clock) continue;
    const ux = dir.x / flat;
    const uy = dir.y / flat;
    const along = (t.x - eye.x) * ux + (t.y - eye.y) * uy;
    if (along <= 0 || along > SHOT_RANGE) continue;
    const off = Math.abs((t.x - eye.x) * uy - (t.y - eye.y) * ux);
    if (off > 0.32) continue;
    // Up and down the aim is forgiving (as in classic shooters), so a villager in a dip or up a slope is hittable.
    const need = Math.atan2(t.z + BODY_MID - eye.z, along);
    if (Math.abs(need - Math.atan2(dir.z, flat)) > AIM_ASSIST_RAD) continue;
    if (along < bestT) {
      best = t;
      bestT = along;
    }
  }
  if (!best) return { fired: true, hit: null, downed: false };
  const wall = wallDistance(
    w,
    eye.x,
    eye.y,
    eye.z,
    best.x - eye.x,
    best.y - eye.y,
    best.z + BODY_MID - eye.z,
    bestT,
  );
  if (wall < bestT || !best.hostile)
    return { fired: true, hit: null, downed: false };
  const e = (c.enemies[best.id] ??= {
    health: ENEMY_HEALTH,
    down: 0,
    cooldown: 0,
  });
  e.health -= SHOT_DAMAGE;
  let downed = false;
  if (e.health <= 0) {
    e.health = ENEMY_HEALTH;
    e.down = c.clock + KNOCKOUT_S;
    downed = true;
  }
  return { fired: true, hit: best.id, downed };
}

export function enemiesAct(
  w: ActionWorld,
  c: Combat,
  player: { x: number; y: number; z: number; h?: number },
  hostiles: readonly Target[],
  safe: (x: number, y: number) => boolean,
): string[] {
  const fired: string[] = [];
  if (safe(player.x, player.y)) return fired;
  for (const h of hostiles) {
    if (!h.hostile) continue;
    const e = (c.enemies[h.id] ??= {
      health: ENEMY_HEALTH,
      down: 0,
      cooldown: 0,
    });
    if (e.down > c.clock || e.cooldown > c.clock || safe(h.x, h.y)) continue;
    const dx = player.x - h.x;
    const dy = player.y - h.y;
    const d = Math.hypot(dx, dy);
    if (d > ENEMY_RANGE) continue;
    const eyeZ = h.z + 1.1;
    const wall = wallDistance(
      w,
      h.x,
      h.y,
      eyeZ,
      dx,
      dy,
      player.z + 0.7 * (player.h ?? STAND_H) - eyeZ,
      d,
    );
    if (wall < d) continue;
    e.cooldown = c.clock + ENEMY_COOLDOWN_S;
    c.health = Math.max(0, c.health - ENEMY_DAMAGE);
    fired.push(h.id);
  }
  return fired;
}

export function tick(c: Combat, dt: number): void {
  c.clock += dt;
  c.cooldown = Math.max(0, c.cooldown - dt);
}

export function respawn(c: Combat): void {
  c.health = MAX_HEALTH;
  c.ammo = START_AMMO;
}

export interface Pickup {
  id: string;
  kind: "health" | "ammo";
  x: number;
  y: number;
}

export function placePickups(
  open: readonly { x: number; y: number }[],
  seed: number,
  count = 8,
): Pickup[] {
  const out: Pickup[] = [];
  if (!open.length) return out;
  let s = seed >>> 0 || 1;
  const next = () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s;
  };
  const used = new Set<number>();
  for (let i = 0; i < count && used.size < open.length; i++) {
    let k = next() % open.length;
    while (used.has(k)) k = (k + 1) % open.length;
    used.add(k);
    out.push({
      id: `pickup-${i}`,
      kind: i % 2 ? "ammo" : "health",
      x: open[k].x + 0.5,
      y: open[k].y + 0.5,
    });
  }
  return out;
}

export function takePickup(
  c: Combat,
  pickups: readonly Pickup[],
  x: number,
  y: number,
): Pickup | null {
  for (const p of pickups) {
    if (c.taken.includes(p.id) || Math.hypot(p.x - x, p.y - y) > 0.6) continue;
    if (p.kind === "health" && c.health >= MAX_HEALTH) continue;
    if (p.kind === "ammo" && c.ammo >= MAX_AMMO) continue;
    c.taken.push(p.id);
    if (p.kind === "health")
      c.health = Math.min(MAX_HEALTH, c.health + PICKUP_HEALTH);
    else c.ammo = Math.min(MAX_AMMO, c.ammo + PICKUP_AMMO);
    return p;
  }
  return null;
}

export function combatOf(raw: unknown): Combat {
  const r = (raw ?? {}) as Partial<Combat>;
  const num = (v: unknown, d: number, max: number) =>
    typeof v === "number" && Number.isFinite(v)
      ? Math.max(0, Math.min(max, v))
      : d;
  const enemies: Combat["enemies"] = {};
  for (const [id, e] of Object.entries(
    r.enemies && typeof r.enemies === "object" ? r.enemies : {},
  ))
    if (e && typeof e === "object")
      enemies[id] = {
        health: num(e.health, ENEMY_HEALTH, ENEMY_HEALTH),
        down: num(e.down, 0, Number.MAX_SAFE_INTEGER),
        cooldown: 0,
      };
  const clock = num(r.clock, 0, Number.MAX_SAFE_INTEGER);
  return {
    health: num(r.health, MAX_HEALTH, MAX_HEALTH) || MAX_HEALTH,
    ammo: num(r.ammo, START_AMMO, MAX_AMMO),
    cooldown: 0,
    enemies,
    taken: (Array.isArray(r.taken) ? r.taken : []).filter(
      (t): t is string => typeof t === "string",
    ),
    clock,
  };
}
