// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Deterministic map generation for the world player: the same WorldSpec (seed + biome + entities) always
// yields the same map, so a shared world looks identical on every machine and saves stay valid.

import { Prng, seedFrom } from "../../expr";
import type { WorldBiome, WorldSpec } from "../../types";

export const TILE_KINDS = [
  "grass",
  "path",
  "water",
  "sand",
  "snow",
  "floor",
  "wall",
  "tree",
  "rock",
  "flowers",
] as const;
export type TileKind = (typeof TILE_KINDS)[number];
export const SOLID: ReadonlySet<TileKind> = new Set<TileKind>([
  "water",
  "wall",
  "tree",
  "rock",
]);
const K = Object.fromEntries(TILE_KINDS.map((k, i) => [k, i])) as Record<
  TileKind,
  number
>;

export interface Point {
  x: number;
  y: number;
}

export interface WorldMap {
  width: number;
  height: number;
  /** Tile kind index (into TILE_KINDS) per cell, row-major. */
  tiles: Uint8Array;
  spawn: Point;
  npcs: Record<string, Point>;
  items: Record<string, Point>;
}

interface BiomeRule {
  ground: TileKind;
  border: TileKind;
  water: number;
  trees: number;
  rocks: number;
  flowers: number;
  rooms?: boolean;
  houses?: boolean;
}

const BIOMES: Record<WorldBiome, BiomeRule> = {
  meadow: {
    ground: "grass",
    border: "tree",
    water: 0.2,
    trees: 0.74,
    rocks: 0.012,
    flowers: 0.08,
  },
  forest: {
    ground: "grass",
    border: "tree",
    water: 0.16,
    trees: 0.6,
    rocks: 0.02,
    flowers: 0.04,
  },
  desert: {
    ground: "sand",
    border: "rock",
    water: 0.1,
    trees: 2,
    rocks: 0.035,
    flowers: 0,
  },
  snow: {
    ground: "snow",
    border: "tree",
    water: 0.16,
    trees: 0.7,
    rocks: 0.02,
    flowers: 0,
  },
  town: {
    ground: "grass",
    border: "tree",
    water: 0.12,
    trees: 0.82,
    rocks: 0.005,
    flowers: 0.05,
    houses: true,
  },
  dungeon: {
    ground: "floor",
    border: "wall",
    water: 0.08,
    trees: 2,
    rocks: 0.02,
    flowers: 0,
    rooms: true,
  },
  beach: {
    ground: "sand",
    border: "water",
    water: 0.36,
    trees: 0.82,
    rocks: 0.01,
    flowers: 0,
  },
};

export const isSolid = (map: WorldMap, x: number, y: number): boolean =>
  x < 0 ||
  y < 0 ||
  x >= map.width ||
  y >= map.height ||
  SOLID.has(TILE_KINDS[map.tiles[y * map.width + x]]);

export const kindAt = (map: WorldMap, x: number, y: number): TileKind =>
  TILE_KINDS[map.tiles[y * map.width + x]];

/** Smooth value noise in [0, 1): random lattice every `cell` tiles, bilinear with smoothstep. */
function valueNoise(
  w: number,
  h: number,
  cell: number,
  rng: Prng,
): Float32Array {
  const gw = Math.ceil(w / cell) + 2;
  const gh = Math.ceil(h / cell) + 2;
  const grid = Float32Array.from({ length: gw * gh }, () => rng.random());
  const out = new Float32Array(w * h);
  const s = (t: number) => t * t * (3 - 2 * t);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
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
      out[y * w + x] =
        (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
    }
  }
  return out;
}

function carve(
  tiles: Uint8Array,
  w: number,
  h: number,
  from: Point,
  to: Point,
  kind: TileKind,
  rng: Prng,
): void {
  let { x, y } = from;
  const set = (px: number, py: number) => {
    if (px > 0 && py > 0 && px < w - 1 && py < h - 1)
      tiles[py * w + px] = K[kind];
  };
  set(x, y);
  let guard = 0;
  while ((x !== to.x || y !== to.y) && guard++ < w * h) {
    const dx = Math.sign(to.x - x);
    const dy = Math.sign(to.y - y);
    // Mostly straight with gentle wiggles so paths look walked, not ruled.
    if (dx !== 0 && (dy === 0 || rng.random() < 0.55)) x += dx;
    else if (dy !== 0) y += dy;
    set(x, y);
  }
}

function placeFree(
  tiles: Uint8Array,
  w: number,
  h: number,
  taken: Point[],
  rng: Prng,
  minDist: number,
  avoid: Point,
): Point {
  for (let attempt = 0; attempt < 400; attempt++) {
    const p = {
      x: 2 + Math.floor(rng.random() * (w - 4)),
      y: 2 + Math.floor(rng.random() * (h - 4)),
    };
    const d = attempt < 300 ? minDist : 2;
    if (SOLID.has(TILE_KINDS[tiles[p.y * w + p.x]]) && attempt < 300) continue;
    if (Math.abs(p.x - avoid.x) + Math.abs(p.y - avoid.y) < d + 2) continue;
    if (taken.some((t) => Math.abs(t.x - p.x) + Math.abs(t.y - p.y) < d))
      continue;
    return p;
  }
  return { x: Math.min(w - 3, avoid.x + 3), y: avoid.y };
}

export function generateMap(spec: WorldSpec): WorldMap {
  const w = Math.max(16, Math.min(256, spec.width | 0));
  const h = Math.max(16, Math.min(256, spec.height | 0));
  const rule = BIOMES[spec.biome] ?? BIOMES.meadow;
  const rng = new Prng(seedFrom("map", spec.seed, spec.biome, w, h));
  const tiles = new Uint8Array(w * h).fill(K[rule.ground]);
  const at = (x: number, y: number) => y * w + x;

  if (rule.rooms) {
    tiles.fill(K.wall);
  } else {
    const water = valueNoise(w, h, 9, rng);
    const trees = valueNoise(w, h, 5, rng);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = at(x, y);
        if (water[i] < rule.water) tiles[i] = K.water;
        else if (water[i] < rule.water + 0.035 && rule.ground !== "sand")
          tiles[i] = K.sand;
        else if (trees[i] > rule.trees) tiles[i] = K.tree;
        else if (rng.random() < rule.rocks) tiles[i] = K.rock;
        else if (rng.random() < rule.flowers) tiles[i] = K.flowers;
      }
    }
  }
  for (let x = 0; x < w; x++) {
    tiles[at(x, 0)] = K[rule.border];
    tiles[at(x, h - 1)] = K[rule.border];
  }
  for (let y = 0; y < h; y++) {
    tiles[at(0, y)] = K[rule.border];
    tiles[at(w - 1, y)] = K[rule.border];
  }

  const spawn = { x: Math.floor(w / 2), y: Math.floor(h / 2) };
  const taken: Point[] = [spawn];
  const npcs: Record<string, Point> = {};
  const items: Record<string, Point> = {};
  const fixed = (p: { x?: number | null; y?: number | null }): Point | null =>
    typeof p.x === "number" &&
    typeof p.y === "number" &&
    p.x > 0 &&
    p.y > 0 &&
    p.x < w - 1 &&
    p.y < h - 1
      ? { x: p.x, y: p.y }
      : null;
  for (const n of spec.npcs) {
    const p = fixed(n) ?? placeFree(tiles, w, h, taken, rng, 7, spawn);
    npcs[n.id] = p;
    taken.push(p);
  }
  for (const it of spec.items) {
    const p = fixed(it) ?? placeFree(tiles, w, h, taken, rng, 5, spawn);
    items[it.id] = p;
    taken.push(p);
  }

  if (rule.rooms) {
    for (const p of taken) {
      const rw = 3 + Math.floor(rng.random() * 4);
      const rh = 2 + Math.floor(rng.random() * 3);
      for (let y = Math.max(1, p.y - rh); y <= Math.min(h - 2, p.y + rh); y++) {
        for (let x = Math.max(1, p.x - rw); x <= Math.min(w - 2, p.x + rw); x++)
          tiles[at(x, y)] = K.floor;
      }
    }
  }
  if (rule.houses) {
    for (let i = 0; i < Math.max(3, Math.floor((w * h) / 400)); i++) {
      const hx = 3 + Math.floor(rng.random() * (w - 10));
      const hy = 3 + Math.floor(rng.random() * (h - 9));
      if (
        taken.some(
          (t) =>
            t.x >= hx - 1 && t.x <= hx + 6 && t.y >= hy - 1 && t.y <= hy + 5,
        )
      )
        continue;
      for (let y = hy; y < hy + 4; y++)
        for (let x = hx; x < hx + 5; x++) tiles[at(x, y)] = K.wall;
      tiles[at(hx + 2, hy + 3)] = K.floor;
    }
  }

  // Paths: spawn to every point of interest, then a ring between neighbours so the map has loops.
  const pathKind: TileKind = rule.rooms ? "floor" : "path";
  for (const p of taken.slice(1)) carve(tiles, w, h, spawn, p, pathKind, rng);
  for (let i = 1; i + 1 < taken.length; i++) {
    if (rng.random() < 0.5)
      carve(tiles, w, h, taken[i], taken[i + 1], pathKind, rng);
  }
  // Clear a little plaza around spawn and every point of interest.
  for (const p of taken) {
    for (let y = p.y - 1; y <= p.y + 1; y++) {
      for (let x = p.x - 1; x <= p.x + 1; x++) {
        if (
          x > 0 &&
          y > 0 &&
          x < w - 1 &&
          y < h - 1 &&
          SOLID.has(TILE_KINDS[tiles[at(x, y)]])
        ) {
          tiles[at(x, y)] = K[rule.rooms ? "floor" : rule.ground];
        }
      }
    }
  }
  return { width: w, height: h, tiles, spawn, npcs, items };
}

/** Tiles reachable from `from` without crossing solid tiles (4-connected). */
export function reachable(map: WorldMap, from: Point): Uint8Array {
  const seen = new Uint8Array(map.width * map.height);
  const queue: number[] = [from.y * map.width + from.x];
  seen[queue[0]] = 1;
  while (queue.length) {
    const i = queue.pop() as number;
    const x = i % map.width;
    const y = (i - x) / map.width;
    for (const [dx, dy] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]) {
      const nx = x + dx;
      const ny = y + dy;
      const j = ny * map.width + nx;
      if (!isSolid(map, nx, ny) && !seen[j]) {
        seen[j] = 1;
        queue.push(j);
      }
    }
  }
  return seen;
}

/** Shortest 4-connected path (BFS) from `from` to the walkable tile nearest `to`; [] when unreachable. */
export function findPath(
  map: WorldMap,
  from: Point,
  to: Point,
  maxNodes = 20_000,
): Point[] {
  const W = map.width;
  let goal = to;
  if (isSolid(map, to.x, to.y)) {
    let best: Point | null = null;
    for (let r = 1; r < 4 && !best; r++) {
      for (let dy = -r; dy <= r && !best; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (!isSolid(map, to.x + dx, to.y + dy)) {
            best = { x: to.x + dx, y: to.y + dy };
            break;
          }
        }
      }
    }
    if (!best) return [];
    goal = best;
  }
  const prev = new Int32Array(map.width * map.height).fill(-1);
  const start = from.y * W + from.x;
  const target = goal.y * W + goal.x;
  prev[start] = start;
  const queue = [start];
  let head = 0;
  while (head < queue.length && head < maxNodes) {
    const i = queue[head++];
    if (i === target) break;
    const x = i % W;
    const y = (i - x) / W;
    for (const [dx, dy] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]) {
      const nx = x + dx;
      const ny = y + dy;
      const j = ny * W + nx;
      if (!isSolid(map, nx, ny) && prev[j] === -1) {
        prev[j] = i;
        queue.push(j);
      }
    }
  }
  if (prev[target] === -1) return [];
  const out: Point[] = [];
  for (let i = target; i !== start; i = prev[i])
    out.push({ x: i % W, y: Math.floor(i / W) });
  return out.reverse();
}
