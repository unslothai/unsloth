// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const P = await import("../src/features/flows/player/world/procgen.ts");
const W = await import("../src/features/flows/player/world/world-state.ts");
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

function spec(seed: number, biome: Spec["biome"], npcs = 5, items = 6): Spec {
  return {
    title: "T",
    seed,
    width: 48,
    height: 36,
    tile_size: 32,
    biome,
    tileset: {},
    player: { name: "Pip" },
    npcs: Array.from({ length: npcs }, (_, i) => ({
      id: `n${i}`,
      name: `N${i}`,
      persona: "",
      greeting: "hi",
    })),
    items: Array.from({ length: items }, (_, i) => ({
      id: `i${i}`,
      name: `I${i}`,
      description: "",
    })),
    quests: [
      {
        id: "q",
        title: "Q",
        description: "",
        giver: "n0",
        requires_item: "i0",
        reward_text: "yay",
      },
    ],
    regions: [],
    vars: {},
    intro_text: "",
  };
}

test("maps are deterministic per seed and differ across seeds", () => {
  const a = P.generateMap(spec(7, "meadow"));
  const b = P.generateMap(spec(7, "meadow"));
  const c = P.generateMap(spec(8, "meadow"));
  assert.deepEqual(a, b);
  assert.notDeepEqual(a.tiles, c.tiles);
});

test("every NPC and item is reachable from spawn in every biome", () => {
  for (const biome of BIOMES) {
    for (let seed = 0; seed < 25; seed++) {
      const s = spec(seed, biome, 1 + (seed % 8), seed % 10);
      const map = P.generateMap(s);
      assert.equal(
        P.isSolid(map, map.spawn.x, map.spawn.y),
        false,
        `${biome}/${seed} spawn`,
      );
      const seen = P.reachable(map, map.spawn);
      for (const [id, p] of [
        ...Object.entries(map.npcs),
        ...Object.entries(map.items),
      ]) {
        assert.equal(
          seen[p.y * map.width + p.x],
          1,
          `${biome}/${seed}: ${id} at ${p.x},${p.y} unreachable`,
        );
      }
      // The border is closed: nobody walks off the map.
      for (let x = 0; x < map.width; x++)
        assert.ok(P.isSolid(map, x, 0) && P.isSolid(map, x, map.height - 1));
    }
  }
});

test("hand-placed entities keep their spot", () => {
  const s = spec(3, "town", 1, 0);
  s.npcs[0] = { ...s.npcs[0], x: 5, y: 6 };
  assert.deepEqual(P.generateMap(s).npcs.n0, { x: 5, y: 6 });
});

test("findPath walks around obstacles and refuses the impossible", () => {
  const map = P.generateMap(spec(11, "forest"));
  const target = map.npcs.n2;
  const path = P.findPath(map, map.spawn, target);
  assert.ok(path.length > 0);
  const last = path[path.length - 1];
  assert.deepEqual(last, target);
  let prev = map.spawn;
  for (const p of path) {
    assert.equal(Math.abs(p.x - prev.x) + Math.abs(p.y - prev.y), 1);
    assert.equal(P.isSolid(map, p.x, p.y), false);
    prev = p;
  }
  const walled = {
    ...map,
    tiles: new Uint8Array(map.tiles.length).fill(P.TILE_KINDS.indexOf("wall")),
  };
  walled.tiles[map.spawn.y * map.width + map.spawn.x] = 0;
  assert.deepEqual(P.findPath(walled, map.spawn, target), []);
});

test("world state: pick up, give, quests and save round-trip", () => {
  const s = spec(1, "meadow");
  let st = W.initialWorldState(s);
  assert.deepEqual(st.inventory, []);
  st = W.pickUp(st, "i0");
  assert.deepEqual(st.inventory, ["i0"]);
  assert.ok(st.picked.includes("i0"));
  st = W.startQuestsOf(s, st, "n0");
  assert.deepEqual(st.activeQuests, ["q"]);
  const { state: after, notes } = W.applyReply(s, st, "n0", {
    text: "Thanks!",
    complete_quest: "q",
    take_item: "i0",
    set: { rep: 1 },
  });
  assert.deepEqual(after.completedQuests, ["q"]);
  assert.deepEqual(after.activeQuests, []);
  assert.deepEqual(after.inventory, []);
  assert.ok(notes.some((n) => n.includes("yay")));
  // Effects the world does not know are ignored even if the server let them through.
  const bogus = W.applyReply(s, after, "n1", {
    text: "x",
    give_item: "dragon",
    complete_quest: "q",
  });
  assert.deepEqual(bogus.state.inventory, []);
  const saved = W.serialize(after);
  assert.deepEqual(W.deserialize(s, JSON.parse(JSON.stringify(saved))), after);
  assert.equal(W.deserialize(s, { v: 99 }), null);
});

test("world state: completing a quest applies its effects", () => {
  const s = spec(1, "meadow");
  s.quests[0].set = { helped: true };
  s.vars = { helped: false };
  let st = W.initialWorldState(s);
  st = W.startQuestsOf(s, W.pickUp(st, "i0"), "n0");
  const { state } = W.applyReply(s, st, "n0", {
    text: "Thanks!",
    complete_quest: "q",
  });
  assert.equal(state.vars.helped, true);
});
