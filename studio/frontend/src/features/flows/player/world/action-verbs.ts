// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { WorldProp3d } from "../../types";
import {
  type ActionWorld,
  type Body,
  type Box,
  type Combat,
  type MoveInput,
  type Target,
  ENEMY_HEALTH,
  KNOCKOUT_S,
  STAND_H,
  STEP_S,
  advance,
  circleHitsBox,
  wallDistance,
} from "./action3d";

export const USABLE_PROPS: ReadonlySet<string> = new Set([
  "door",
  "chest",
  "lever",
]);
export const isUsable = (p: WorldProp3d) => USABLE_PROPS.has(p.kind);

export interface PropFlags {
  open?: boolean;
  unlocked?: boolean;
  looted?: boolean;
}
export type PropStates = Record<string, PropFlags>;

export const isOpen = (p: WorldProp3d, st: PropStates) =>
  st[p.id]?.open ?? p.open === true;

export const closedProps = (
  props: readonly WorldProp3d[],
  st: PropStates,
): WorldProp3d[] => props.filter((p) => !(p.kind === "door" && isOpen(p, st)));

/** Items still waiting in unopened chests; they are not on the ground. */
export function heldInChests(
  props: readonly WorldProp3d[],
  st: PropStates,
): string[] {
  const out: string[] = [];
  for (const p of props)
    if (p.kind === "chest" && !st[p.id]?.looted)
      for (const id of p.contents ?? []) out.push(id);
  return out;
}

export function propStatesOf(
  raw: unknown,
  props: readonly WorldProp3d[],
): PropStates {
  const out: PropStates = {};
  if (!raw || typeof raw !== "object") return out;
  const r = raw as Record<string, unknown>;
  for (const p of props) {
    const v = r[p.id];
    if (!isUsable(p) || !v || typeof v !== "object") continue;
    const f: PropFlags = {};
    for (const k of ["open", "unlocked", "looted"] as const)
      if (typeof (v as PropFlags)[k] === "boolean") f[k] = (v as PropFlags)[k];
    if (Object.keys(f).length) out[p.id] = f;
  }
  return out;
}

export interface UseOutcome {
  states: PropStates;
  gained: string[];
  changed: string[];
  text: string;
  loud: boolean;
}

export function interactProp(
  prop: WorldProp3d,
  props: readonly WorldProp3d[],
  states: PropStates,
  inventory: readonly string[],
  opts: {
    name?: (itemId: string) => string;
    occupied?: (p: WorldProp3d) => boolean;
  } = {},
): UseOutcome {
  const name = opts.name ?? ((id: string) => id);
  const st: PropStates = { ...states };
  const set = (id: string, f: PropFlags) => {
    st[id] = { ...st[id], ...f };
  };
  const out: UseOutcome = {
    states: st,
    gained: [],
    changed: [],
    text: "",
    loud: false,
  };
  const toggle = (door: WorldProp3d): boolean => {
    const open = isOpen(door, st);
    if (open && opts.occupied?.(door)) return false;
    set(door.id, { open: !open });
    out.changed.push(door.id);
    return true;
  };
  if (prop.kind === "door") {
    const key = prop.locked_by;
    if (key && !st[prop.id]?.unlocked && !isOpen(prop, st)) {
      if (!inventory.includes(key)) {
        out.text = `Locked. It needs the ${name(key)}`;
        return out;
      }
      set(prop.id, { unlocked: true });
      out.text = `Unlocked with the ${name(key)}`;
    }
    if (!toggle(prop)) out.text = "Something is in the way";
    else {
      out.loud = true;
      out.text ||= isOpen(prop, st) ? "The door opens" : "The door closes";
    }
  } else if (prop.kind === "chest") {
    if (st[prop.id]?.looted) out.text = "The chest is empty";
    else {
      set(prop.id, { looted: true, open: true });
      out.changed.push(prop.id);
      out.gained = (prop.contents ?? []).filter(
        (id, i, all) => !inventory.includes(id) && all.indexOf(id) === i,
      );
      out.text = out.gained.length
        ? `Found ${out.gained.map(name).join(", ")}`
        : "The chest is empty";
    }
  } else if (prop.kind === "lever") {
    set(prop.id, { open: !isOpen(prop, st) });
    out.changed.push(prop.id);
    let moved = 0;
    let stuck = 0;
    for (const id of prop.targets ?? []) {
      const door = props.find((p) => p.id === id && p.kind === "door");
      if (!door) continue;
      if (toggle(door)) {
        moved++;
        if (door.locked_by) set(door.id, { unlocked: true });
      } else stuck++;
    }
    out.loud = moved > 0;
    out.text = stuck
      ? "The lever clicks, but something blocks a door"
      : moved
        ? "The lever clicks and a door moves"
        : "The lever clicks";
  }
  return out;
}

export const MELEE_RANGE = 1.2;
export const MELEE_HALF_WIDTH = 0.5;
/** How far above or below a hostile's body the strike height may be and still connect. */
export const MELEE_REACH_Z = 0.5;
export const MELEE_COOLDOWN_S = 0.45;
export const MELEE_DAMAGE = 50;
export const KNOCKBACK = 0.9;
export const DASH_S = 0.18;
export const DASH_SPEED = 14;
export const DASH_DISTANCE = DASH_S * DASH_SPEED;
export const DASH_COOLDOWN_S = 0.8;
export const THROW_SPEED = 9;
export const THROW_LIFT = 2.5;
export const THROW_FUSE_S = 1.4;
export const THROW_COOLDOWN_S = 0.8;
export const THROW_AMMO = 3;
export const SPLASH_RADIUS = 2.4;
export const SPLASH_DAMAGE = 70;
export const PROJECTILE_R = 0.09;
const GRAVITY = 16;
const MAX_STEPS = 6;
const SUB_MOVE = 0.08;
const MAX_PROJECTILES = 4;
const MAX_NOISES = 16;
const BODY_MID = 0.6;

/** Noise radius in tiles by source; hostile AI hears a noise inside its radius, fading with distance. */
export const NOISE = {
  shot: 16,
  burst: 14,
  door: 9,
  sprint: 6,
  walk: 2.5,
  crouch: 0.8,
} as const;

/** How far a footstep carries: crouching is near silent, sprinting loud. */
export const stepNoise = (b: Pick<Body, "crouched">, sprint: boolean) =>
  b.crouched ? NOISE.crouch : sprint ? NOISE.sprint : NOISE.walk;

export interface Projectile {
  live: boolean;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  fuse: number;
}

export interface Noise {
  seq: number;
  x: number;
  y: number;
  radius: number;
  at: number;
}

export interface Verbs {
  clock: number;
  meleeReady: number;
  dashReady: number;
  throwReady: number;
  dashLeft: number;
  dashSteps: number;
  dashInput: MoveInput;
  acc: number;
  shots: Projectile[];
  live: number;
  noises: Noise[];
  noiseSeq: number;
}

export const newVerbs = (): Verbs => ({
  clock: 0,
  meleeReady: 0,
  dashReady: 0,
  throwReady: 0,
  dashLeft: 0,
  dashSteps: 0,
  dashInput: { dx: 0, dy: 0, sprint: false, crouch: false, jump: false },
  acc: 0,
  shots: Array.from({ length: MAX_PROJECTILES }, () => ({
    live: false,
    x: 0,
    y: 0,
    z: 0,
    vx: 0,
    vy: 0,
    vz: 0,
    fuse: 0,
  })),
  live: 0,
  noises: Array.from({ length: MAX_NOISES }, () => ({
    seq: 0,
    x: 0,
    y: 0,
    radius: 0,
    at: 0,
  })),
  noiseSeq: 0,
});

export function verbsTick(v: Verbs, dt: number): void {
  v.clock += dt;
}

export function makeNoise(
  v: Verbs,
  x: number,
  y: number,
  radius: number,
): void {
  const n = v.noises[v.noiseSeq % MAX_NOISES];
  v.noiseSeq++;
  n.seq = v.noiseSeq;
  n.x = x;
  n.y = y;
  n.radius = radius;
  n.at = v.clock;
}

export function hurt(c: Combat, id: string, damage: number): boolean {
  const e = (c.enemies[id] ??= {
    health: ENEMY_HEALTH,
    down: 0,
    cooldown: 0,
  });
  if (e.down > c.clock) return false;
  e.health -= damage;
  if (e.health > 0) return false;
  e.health = ENEMY_HEALTH;
  e.down = c.clock + KNOCKOUT_S;
  return true;
}

export interface MeleeHit {
  hit: string | null;
  downed: boolean;
  kx: number;
  ky: number;
}

/** A short box in front of the player: the nearest hostile inside it takes damage and is pushed back. */
export function melee(
  w: ActionWorld,
  c: Combat,
  v: Verbs,
  eye: { x: number; y: number; z: number },
  fx: number,
  fy: number,
  targets: readonly Target[],
): MeleeHit | null {
  if (v.clock < v.meleeReady) return null;
  v.meleeReady = v.clock + MELEE_COOLDOWN_S;
  const len = Math.hypot(fx, fy) || 1;
  const ux = fx / len;
  const uy = fy / len;
  let best: Target | null = null;
  let bestAlong = Number.POSITIVE_INFINITY;
  for (const t of targets) {
    if (!t.hostile || (c.enemies[t.id]?.down ?? 0) > c.clock) continue;
    const along = (t.x - eye.x) * ux + (t.y - eye.y) * uy;
    const side = Math.abs((t.x - eye.x) * uy - (t.y - eye.y) * ux);
    if (along <= 0 || along > MELEE_RANGE || side > MELEE_HALF_WIDTH) continue;
    if (eye.z < t.z - MELEE_REACH_Z || eye.z > t.z + STAND_H + MELEE_REACH_Z)
      continue;
    if (along < bestAlong) {
      best = t;
      bestAlong = along;
    }
  }
  if (!best) return { hit: null, downed: false, kx: 0, ky: 0 };
  const d = Math.hypot(best.x - eye.x, best.y - eye.y);
  const wall = wallDistance(
    w,
    eye.x,
    eye.y,
    eye.z,
    best.x - eye.x,
    best.y - eye.y,
    best.z + BODY_MID - eye.z,
    d,
  );
  if (wall < d - 0.2) return { hit: null, downed: false, kx: 0, ky: 0 };
  const downed = hurt(c, best.id, MELEE_DAMAGE);
  return { hit: best.id, downed, kx: ux * KNOCKBACK, ky: uy * KNOCKBACK };
}

/** Starts a dash along (dx, dy) (the facing when there is no input); false while cooling down or crouched. */
export function startDash(v: Verbs, b: Body, dx: number, dy: number): boolean {
  const len = Math.hypot(dx, dy);
  if (v.clock < v.dashReady || b.crouched || len < 1e-6) return false;
  v.dashReady = v.clock + DASH_S + DASH_COOLDOWN_S;
  v.dashLeft = DASH_DISTANCE;
  v.dashSteps = Math.ceil(DASH_S / STEP_S - 1e-9) + 1;
  v.dashInput.dx = dx / len;
  v.dashInput.dy = dy / len;
  return true;
}

export const dashing = (v: Verbs) => v.dashSteps > 0;

/** One frame of a dash: the body's own fixed steps at dash speed, capped to the dash distance. */
export function dashAdvance(
  w: ActionWorld,
  b: Body,
  v: Verbs,
  dt: number,
): number {
  const input = v.dashInput;
  input.limit = v.dashLeft;
  const n = advance(w, b, input, dt, DASH_SPEED);
  v.dashLeft = Math.max(0, v.dashLeft - n * DASH_SPEED * STEP_S);
  v.dashSteps = v.dashLeft > 1e-9 ? Math.max(0, v.dashSteps - n) : 0;
  return n;
}

export function throwRock(
  c: Combat,
  v: Verbs,
  eye: { x: number; y: number; z: number },
  dir: { x: number; y: number; z: number },
): boolean {
  if (v.clock < v.throwReady || c.ammo < THROW_AMMO) return false;
  const p = v.shots.find((s) => !s.live);
  if (!p) return false;
  const len = Math.hypot(dir.x, dir.y, dir.z) || 1;
  c.ammo -= THROW_AMMO;
  v.throwReady = v.clock + THROW_COOLDOWN_S;
  p.live = true;
  p.x = eye.x + (dir.x / len) * 0.3;
  p.y = eye.y + (dir.y / len) * 0.3;
  p.z = eye.z - 0.15;
  p.vx = (dir.x / len) * THROW_SPEED;
  p.vy = (dir.y / len) * THROW_SPEED;
  p.vz = (dir.z / len) * THROW_SPEED + THROW_LIFT;
  p.fuse = THROW_FUSE_S;
  v.live++;
  return true;
}

function boxAt(
  w: ActionWorld,
  x: number,
  y: number,
  z: number,
  r: number,
): Box | null {
  for (let cy = Math.floor(y - r); cy <= Math.floor(y + r); cy++)
    for (let cx = Math.floor(x - r); cx <= Math.floor(x + r); cx++) {
      const list = w.cells.get(cy * w.width + cx);
      if (!list) continue;
      for (const b of list)
        if (z + r > b.bottom && z - r < b.top && circleHitsBox(b, x, y, r))
          return b;
    }
  return null;
}

function floorUnder(
  w: ActionWorld,
  x: number,
  y: number,
  z: number,
  r: number,
): number {
  let floor = w.ground(x, y);
  for (let cy = Math.floor(y - r); cy <= Math.floor(y + r); cy++)
    for (let cx = Math.floor(x - r); cx <= Math.floor(x + r); cx++) {
      const list = w.cells.get(cy * w.width + cx);
      if (!list) continue;
      for (const b of list)
        if (b.top > floor && b.top <= z && circleHitsBox(b, x, y, r))
          floor = b.top;
    }
  return floor;
}

const blockedAt = (
  w: ActionWorld,
  x: number,
  y: number,
  z: number,
  r: number,
) => w.solidTile(Math.floor(x), Math.floor(y)) || boxAt(w, x, y, z, r) !== null;

function burst(v: Verbs, p: Projectile, onBurst: (p: Projectile) => void) {
  p.live = false;
  v.live--;
  makeNoise(v, p.x, p.y, NOISE.burst);
  onBurst(p);
}

/**
 * Thrown rocks at the fixed step: gravity, bounces off walls and boxes, a fuse; a rock bursts early when it
 * reaches a hostile (`hits`), and `onBurst` sees each burst once.
 */
export function stepProjectiles(
  w: ActionWorld,
  v: Verbs,
  dt: number,
  hits: (x: number, y: number, z: number) => boolean,
  onBurst: (p: Projectile) => void,
): void {
  if (!v.live) {
    v.acc = 0;
    return;
  }
  v.acc = Math.min(v.acc + dt, MAX_STEPS * STEP_S);
  const r = PROJECTILE_R;
  while (v.acc >= STEP_S - 1e-9) {
    v.acc -= STEP_S;
    for (const p of v.shots) {
      if (!p.live) continue;
      p.fuse -= STEP_S;
      if (p.fuse <= 0) {
        burst(v, p, onBurst);
        continue;
      }
      p.vz -= GRAVITY * STEP_S;
      const subs = Math.max(
        1,
        Math.ceil((Math.hypot(p.vx, p.vy, p.vz) * STEP_S) / SUB_MOVE),
      );
      const h = STEP_S / subs;
      for (let i = 0; i < subs && p.live; i++) {
        const nx = p.x + p.vx * h;
        const ny = p.y + p.vy * h;
        if (blockedAt(w, nx, ny, p.z, r)) {
          const bx = blockedAt(w, nx, p.y, p.z, r);
          const by = blockedAt(w, p.x, ny, p.z, r);
          if (bx || !by) p.vx *= -0.35;
          if (by || !bx) p.vy *= -0.35;
        } else {
          p.x = nx;
          p.y = ny;
        }
        const nz = p.z + p.vz * h;
        const floor = floorUnder(w, p.x, p.y, p.z, r);
        if (nz - r <= floor) {
          p.z = floor + r;
          p.vz = p.vz < -1.5 ? -p.vz * 0.3 : 0;
          p.vx *= 0.6;
          p.vy *= 0.6;
        } else {
          const roof = p.vz > 0 ? boxAt(w, p.x, p.y, nz, r) : null;
          if (roof) p.vz = 0;
          else p.z = nz;
        }
        if (hits(p.x, p.y, p.z)) burst(v, p, onBurst);
      }
    }
  }
}

export function splash(
  w: ActionWorld,
  c: Combat,
  at: { x: number; y: number; z: number },
  targets: readonly Target[],
): { id: string; downed: boolean }[] {
  const out: { id: string; downed: boolean }[] = [];
  const oz = Math.max(at.z, w.ground(at.x, at.y) + 0.3);
  for (const t of targets) {
    if (!t.hostile || (c.enemies[t.id]?.down ?? 0) > c.clock) continue;
    const d = Math.hypot(t.x - at.x, t.y - at.y);
    // Distance to the nearest point of the body, so a burst far above or below falls off too.
    const dz = Math.max(0, t.z - oz, oz - (t.z + STAND_H));
    const r = Math.hypot(d, dz);
    if (r > SPLASH_RADIUS) continue;
    if (
      d > 0.3 &&
      wallDistance(
        w,
        at.x,
        at.y,
        oz,
        t.x - at.x,
        t.y - at.y,
        t.z + BODY_MID - oz,
        d,
      ) <
        d - 0.25
    )
      continue;
    const dmg = Math.round(SPLASH_DAMAGE * (1 - r / SPLASH_RADIUS));
    if (dmg > 0) out.push({ id: t.id, downed: hurt(c, t.id, dmg) });
  }
  return out;
}
