// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const P = await import("../src/features/flows/player/world/procgen.ts");
const V = await import("../src/features/flows/player/world/world3d.ts");
type Spec = import("../src/features/flows/types.ts").WorldSpec;

const BIOMES = [
  "meadow",
  "forest",
  "desert",
  "snow",
  "town",
  "dungeon",
  "beach",
] as const;

function spec(
  seed: number,
  biome: Spec["biome"],
  extra: Partial<Spec> = {},
): Spec {
  return {
    title: "T",
    seed,
    width: 48,
    height: 36,
    tile_size: 32,
    biome,
    tileset: {},
    player: { name: "You" },
    npcs: Array.from({ length: 5 }, (_, i) => ({
      id: `n${i}`,
      name: `N${i}`,
      persona: "",
      greeting: "hi",
    })),
    items: Array.from({ length: 6 }, (_, i) => ({
      id: `i${i}`,
      name: `I${i}`,
      description: "",
    })),
    quests: [],
    regions: [],
    vars: {},
    intro_text: "",
    ...extra,
  } as Spec;
}

test("view settings fill defaults, clamp and pick the sky from the time of day", () => {
  const d = V.resolveView3d(spec(1, "meadow"));
  assert.deepEqual(d, {
    time_of_day: "day",
    sky: V.TIMES.day.sky,
    fog: 0.35,
    camera_height: 0.85,
    walk_speed: 4.4,
    terrain: 1,
  });
  const night = V.resolveView3d({ view3d: { time_of_day: "night" } });
  assert.equal(night.sky, V.TIMES.night.sky);
  const odd = V.resolveView3d({
    view3d: {
      time_of_day: "noon" as never,
      sky: "blue",
      fog: 9,
      camera_height: -1,
      walk_speed: Number.NaN,
      terrain: 0,
    },
  });
  assert.equal(odd.time_of_day, "day");
  assert.equal(odd.sky, V.TIMES.day.sky);
  assert.equal(odd.fog, 1);
  assert.equal(odd.camera_height, 0.3);
  assert.equal(odd.walk_speed, 4.4);
  assert.equal(odd.terrain, 0);
  assert.equal(V.resolveView3d({ view3d: { sky: "#AABBCC" } }).sky, "#aabbcc");
  const [near, far] = V.fogRange(1);
  assert.ok(near < far && far < V.fogRange(0)[1]);
});

test("every tile kind has a colour and a lift, and water sits below the ground", () => {
  for (const k of P.TILE_KINDS) {
    assert.match(V.TILE_COLORS[k], /^#[0-9a-f]{6}$/);
    assert.equal(typeof V.TILE_LIFT[k], "number");
  }
  for (const k of P.TILE_KINDS)
    if (k !== "water") assert.ok(V.TILE_LIFT.water < V.TILE_LIFT[k], k);
  const map = P.generateMap(spec(3, "beach"));
  for (let y = 0; y < map.height; y++)
    for (let x = 0; x < map.width; x++) {
      const look = V.tileLook(map, x, y);
      assert.equal(look.kind, P.kindAt(map, x, y));
      assert.equal(look.color, V.TILE_COLORS[look.kind]);
    }
});

test("terrain is deterministic from the seed and dips under water", () => {
  for (const biome of BIOMES) {
    const s = spec(11, biome);
    const map = P.generateMap(s);
    const a = V.buildTerrain(map, s.seed, 1);
    const b = V.buildTerrain(P.generateMap(s), s.seed, 1);
    assert.deepEqual(a.corners, b.corners, biome);
    assert.equal(a.corners.length, (map.width + 1) * (map.height + 1));
    assert.ok(a.corners.every(Number.isFinite));
    const other = V.buildTerrain(map, s.seed + 1, 1);
    assert.notDeepEqual(a.corners, other.corners, biome);
    for (let y = 1; y < map.height - 1; y++)
      for (let x = 1; x < map.width - 1; x++) {
        if (P.kindAt(map, x, y) !== "water") continue;
        const flat = V.buildTerrain(map, s.seed, 0);
        assert.ok(V.heightAt(flat, x + 0.5, y + 0.5) < 0, `${biome} ${x},${y}`);
        assert.ok(V.waterLevel(a, x, y) > V.heightAt(a, x + 0.5, y + 0.5));
        y = map.height;
        break;
      }
  }
  const s = spec(5, "dungeon");
  const map = P.generateMap(s);
  const flat = V.buildTerrain(map, s.seed, 0);
  assert.equal(V.heightAt(flat, map.spawn.x + 0.5, map.spawn.y + 0.5), 0);
  const m = P.generateMap(spec(5, "meadow"));
  const lo = V.buildTerrain(m, 5, 0.5).corners;
  const hi = V.buildTerrain(m, 5, 2).corners;
  const range = (c: Float32Array) => Math.max(...c) - Math.min(...c);
  assert.ok(range(hi) > range(lo));
});

test("height is continuous across tile edges", () => {
  const s = spec(7, "meadow");
  const t = V.buildTerrain(P.generateMap(s), s.seed, 1.5);
  for (let x = 1; x < 40; x++) {
    const left = V.heightAt(t, x - 1e-4, 10.3);
    const right = V.heightAt(t, x + 1e-4, 10.3);
    assert.ok(Math.abs(left - right) < 1e-2, `x=${x}`);
  }
});

test("every NPC, item and the spawn stand where the 3D player can walk", () => {
  for (const biome of BIOMES) {
    for (const seed of [1, 2, 3, 99]) {
      const s = spec(seed, biome, {
        regions: [
          { id: "a", name: "A", x: 2, y: 2, w: 12, h: 10, on_enter_text: "" },
          { id: "b", name: "B", x: 30, y: 20, w: 14, h: 12, on_enter_text: "" },
        ],
      });
      const map = P.generateMap(s);
      const spots = [
        map.spawn,
        ...Object.values(map.npcs),
        ...Object.values(map.items),
      ];
      for (const p of spots) {
        assert.ok(!P.isSolid(map, p.x, p.y), `${biome}/${seed} ${p.x},${p.y}`);
        assert.ok(
          V.canStand(map, p.x + 0.5, p.y + 0.5),
          `${biome}/${seed} ${p.x},${p.y}`,
        );
      }
      // Houses replace a tree or rock, never a walkable cell or the border ring.
      const taken = new Set(spots.map((p) => `${p.x},${p.y}`));
      for (const h of V.houseCells(s, map)) {
        assert.ok(
          ["tree", "rock"].includes(P.kindAt(map, h.x, h.y)),
          `${biome}/${seed}`,
        );
        assert.ok(!taken.has(`${h.x},${h.y}`));
        assert.ok(
          h.x > 0 && h.y > 0 && h.x < map.width - 1 && h.y < map.height - 1,
        );
      }
    }
  }
});

test("walking slides along walls and never enters a solid tile", () => {
  const s = spec(4, "town");
  const map = P.generateMap(s);
  let x = map.spawn.x + 0.5;
  let y = map.spawn.y + 0.5;
  for (let i = 0; i < 4000; i++) {
    const a = (i * 2.399) % (Math.PI * 2);
    [x, y] = V.slide(map, x, y, Math.cos(a) * 0.2, Math.sin(a) * 0.2);
    assert.ok(V.canStand(map, x, y), `step ${i}`);
  }
  let [bx, by] = [map.spawn.x + 0.5, map.spawn.y + 0.5];
  for (let i = 0; i < 400; i++) [bx, by] = V.slide(map, bx, by, -0.2, 0);
  assert.ok(bx > 1 && V.canStand(map, bx, by));
});

test("decor lists match the map", () => {
  const s = spec(8, "forest");
  const map = P.generateMap(s);
  const d = V.decorCells(map);
  for (const k of ["tree", "rock", "wall", "flowers"] as const) {
    let n = 0;
    for (let i = 0; i < map.tiles.length; i++)
      if (P.TILE_KINDS[map.tiles[i]] === k) n++;
    assert.equal(d[k].length, n, k);
    for (const p of d[k]) assert.equal(P.kindAt(map, p.x, p.y), k);
  }
});
