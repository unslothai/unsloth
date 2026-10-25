// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No three.js here (runs under node --test). Tile (x, y) covers [x, x + 1] x [y, y + 1]; tile y is world z.

import { Prng, seedFrom } from "../../expr";
import type { World3dView, WorldSpec } from "../../types";
import {
  type Point,
  TILE_KINDS,
  type TileKind,
  type WorldMap,
  isSolid,
  kindAt,
} from "./procgen";

export type TimeOfDay = NonNullable<World3dView["time_of_day"]>;

export interface View3d {
  time_of_day: TimeOfDay;
  sky: string;
  fog: number;
  camera_height: number;
  walk_speed: number;
  terrain: number;
}

export interface Lighting {
  sky: string;
  hemi: [string, string, number];
  sun: [string, number, number];
}

export const TIMES: Record<TimeOfDay, Lighting> = {
  dawn: {
    sky: "#f4b9a0",
    hemi: ["#ffe2cf", "#6a5a70", 2.3],
    sun: ["#ffc58a", 2.0, 14],
  },
  day: {
    sky: "#8ecbf0",
    hemi: ["#eef7ff", "#7d8a62", 2.6],
    sun: ["#fff6e0", 2.6, 55],
  },
  dusk: {
    sky: "#e48a6a",
    hemi: ["#ffd0b0", "#5a4a68", 2.1],
    sun: ["#ffa060", 1.7, 10],
  },
  night: {
    sky: "#16203a",
    hemi: ["#8095d0", "#202638", 1.5],
    sun: ["#b4c4ff", 0.7, 40],
  },
};

export const VIEW3D_DEFAULTS: Omit<View3d, "sky"> = {
  time_of_day: "day",
  fog: 0.35,
  camera_height: 0.85,
  walk_speed: 4.4,
  terrain: 1,
};

const HEX = /^#[0-9a-f]{6}$/i;

const clamp = (v: unknown, lo: number, hi: number, d: number): number =>
  typeof v === "number" && Number.isFinite(v)
    ? Math.max(lo, Math.min(hi, v))
    : d;

export function resolveView3d(spec: Pick<WorldSpec, "view3d"> | null): View3d {
  const v = spec?.view3d ?? {};
  const timeOfDay: TimeOfDay =
    v.time_of_day && v.time_of_day in TIMES
      ? v.time_of_day
      : VIEW3D_DEFAULTS.time_of_day;
  return {
    time_of_day: timeOfDay,
    sky:
      typeof v.sky === "string" && HEX.test(v.sky)
        ? v.sky.toLowerCase()
        : TIMES[timeOfDay].sky,
    fog: clamp(v.fog, 0, 1, VIEW3D_DEFAULTS.fog),
    camera_height: clamp(
      v.camera_height,
      0.3,
      4,
      VIEW3D_DEFAULTS.camera_height,
    ),
    walk_speed: clamp(v.walk_speed, 0.5, 20, VIEW3D_DEFAULTS.walk_speed),
    terrain: clamp(v.terrain, 0, 3, VIEW3D_DEFAULTS.terrain),
  };
}

/** Fog start and end distance in tiles: 0 is clear to the far edge of a big map, 1 is a short misty walk. */
export function fogRange(fog: number): [number, number] {
  const far = 140 - 120 * fog;
  return [far * 0.25, far];
}

export const TILE_COLORS: Record<TileKind, string> = {
  grass: "#6fae4f",
  path: "#c9a874",
  water: "#3f7fb8",
  sand: "#e3cf96",
  snow: "#eef3f8",
  floor: "#8a8072",
  wall: "#6b6672",
  tree: "#4f8f3d",
  rock: "#8d8a83",
  flowers: "#79b85a",
};

export const TILE_LIFT: Record<TileKind, number> = {
  grass: 0,
  path: -0.02,
  water: -0.45,
  sand: -0.04,
  snow: 0,
  floor: 0,
  wall: 0,
  tree: 0.04,
  rock: 0.06,
  flowers: 0,
};

export function tileLook(
  map: WorldMap,
  x: number,
  y: number,
): { kind: TileKind; color: string; lift: number } {
  const kind = kindAt(map, x, y);
  return { kind, color: TILE_COLORS[kind], lift: TILE_LIFT[kind] };
}

export interface Terrain {
  width: number;
  height: number;
  corners: Float32Array;
}

export function buildTerrain(
  map: WorldMap,
  seed: number,
  scale: number,
): Terrain {
  const W = map.width;
  const H = map.height;
  const cell = 7;
  const gw = Math.ceil(W / cell) + 2;
  const gh = Math.ceil(H / cell) + 2;
  const rng = new Prng(seedFrom("terrain3d", seed, W, H));
  const grid = Float32Array.from({ length: gw * gh }, () => rng.random());
  const s = (t: number) => t * t * (3 - 2 * t);
  const hill = (x: number, y: number) => {
    const gx = x / cell;
    const gy = y / cell;
    const x0 = Math.floor(gx);
    const y0 = Math.floor(gy);
    const tx = s(gx - x0);
    const ty = s(gy - y0);
    const a = grid[y0 * gw + x0];
    const b = grid[y0 * gw + x0 + 1];
    const c = grid[(y0 + 1) * gw + x0];
    const d = grid[(y0 + 1) * gw + x0 + 1];
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
  };
  const corners = new Float32Array((W + 1) * (H + 1));
  for (let y = 0; y <= H; y++) {
    for (let x = 0; x <= W; x++) {
      let lift = 0;
      let n = 0;
      let water = false;
      for (const [tx, ty] of [
        [x - 1, y - 1],
        [x, y - 1],
        [x - 1, y],
        [x, y],
      ]) {
        if (tx < 0 || ty < 0 || tx >= W || ty >= H) continue;
        const k = kindAt(map, tx, ty);
        water ||= k === "water";
        lift += TILE_LIFT[k];
        n++;
      }
      const base = (hill(x, y) - 0.5) * 1.6 * scale;
      corners[y * (W + 1) + x] =
        base + (water ? TILE_LIFT.water : lift / Math.max(1, n));
    }
  }
  return { width: W, height: H, corners };
}

export function heightAt(t: Terrain, x: number, y: number): number {
  const cx = Math.max(0, Math.min(t.width - 1e-6, x));
  const cy = Math.max(0, Math.min(t.height - 1e-6, y));
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const fx = cx - x0;
  const fy = cy - y0;
  const r = t.width + 1;
  const a = t.corners[y0 * r + x0];
  const b = t.corners[y0 * r + x0 + 1];
  const c = t.corners[(y0 + 1) * r + x0];
  const d = t.corners[(y0 + 1) * r + x0 + 1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

export function waterLevel(t: Terrain, x: number, y: number): number {
  return heightAt(t, x + 0.5, y + 0.5) - TILE_LIFT.water - 0.18;
}

export function decorCells(
  map: WorldMap,
): Record<"tree" | "rock" | "wall" | "flowers", Point[]> {
  const out: Record<"tree" | "rock" | "wall" | "flowers", Point[]> = {
    tree: [],
    rock: [],
    wall: [],
    flowers: [],
  };
  for (let y = 0; y < map.height; y++) {
    for (let x = 0; x < map.width; x++) {
      const k = TILE_KINDS[map.tiles[y * map.width + x]];
      if (k === "tree" || k === "rock" || k === "wall" || k === "flowers")
        out[k].push({ x, y });
    }
  }
  return out;
}

/** One cottage per region, only on a solid cell so walking and NPC paths match the 2D map exactly. */
export function houseCells(
  spec: Pick<WorldSpec, "regions">,
  map: WorldMap,
): Point[] {
  const used = new Set<string>();
  const out: Point[] = [];
  for (const r of spec.regions) {
    const cx = Math.floor(r.x + r.w / 2);
    const cy = Math.floor(r.y + r.h / 2);
    let best: Point | null = null;
    let bestD = Number.POSITIVE_INFINITY;
    for (let y = cy - 4; y <= cy + 4; y++) {
      for (let x = cx - 4; x <= cx + 4; x++) {
        // Never on the border ring, so the map edge stays a wall of trees.
        if (x < 1 || y < 1 || x >= map.width - 1 || y >= map.height - 1)
          continue;
        const k = kindAt(map, x, y);
        if (k !== "tree" && k !== "rock") continue;
        const d = (x - cx) ** 2 + (y - cy) ** 2;
        if (d < bestD && !used.has(`${x},${y}`)) {
          best = { x, y };
          bestD = d;
        }
      }
    }
    if (best) {
      used.add(`${best.x},${best.y}`);
      out.push(best);
    }
  }
  return out;
}

/** True when a circle of `radius` tiles at (x, y) overlaps no solid tile: the walk collision test. */
export function canStand(
  map: WorldMap,
  x: number,
  y: number,
  radius = 0.22,
): boolean {
  for (const [dx, dy] of [
    [-radius, -radius],
    [radius, -radius],
    [-radius, radius],
    [radius, radius],
  ]) {
    if (isSolid(map, Math.floor(x + dx), Math.floor(y + dy))) return false;
  }
  return true;
}

export function slide(
  map: WorldMap,
  x: number,
  y: number,
  dx: number,
  dy: number,
): [number, number] {
  let nx = x;
  let ny = y;
  if (canStand(map, x + dx, y)) nx = x + dx;
  if (canStand(map, nx, y + dy)) ny = y + dy;
  return [nx, ny];
}

let webgl: boolean | null = null;

export function webglAvailable(): boolean {
  if (webgl !== null) return webgl;
  try {
    const c = document.createElement("canvas");
    const gl = (c.getContext("webgl2") ??
      c.getContext("webgl")) as WebGLRenderingContext | null;
    webgl = Boolean(gl);
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
  } catch {
    webgl = false;
  }
  return webgl;
}
