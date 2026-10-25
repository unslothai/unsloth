// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const W = await import("../src/features/flows/player/world/world-state.ts");
const D = await import("../src/features/flows/player/world/world-director.ts");
const P = await import("../src/features/flows/player/world/procgen.ts");
type Spec = import("../src/features/flows/types.ts").WorldSpec;
type Patch = Parameters<typeof D.applyPatch>[3];

const SPEC = {
  title: "Lantern Village",
  seed: 7,
  width: 32,
  height: 24,
  tile_size: 32,
  biome: "meadow",
  tileset: {},
  player: { name: "Pip" },
  npcs: [{ id: "tom", name: "Old Tom", persona: "", greeting: "hi" }],
  items: [{ id: "rod", name: "Fishing Rod", description: "" }],
  quests: [
    {
      id: "lost_rod",
      title: "The Lost Rod",
      description: "",
      giver: "tom",
      requires_item: "rod",
      reward_text: "Thanks",
    },
  ],
  regions: [
    { id: "pond", name: "Pond", x: 2, y: 2, w: 8, h: 6, on_enter_text: "" },
  ],
  vars: {},
  intro_text: "",
} as unknown as Spec;

const PATCH: Patch = {
  quest: {
    id: "ring_bell",
    title: "Ring the Bell",
    description: "",
    giver: "mira",
    requires_item: "bell",
    reward_text: "Lovely",
  },
  npcs: [
    {
      id: "mira",
      name: "Mira",
      persona: "kind",
      greeting: "Hello",
      region: "pond",
    },
  ],
  items: [{ id: "bell", name: "Golden Bell", description: "", region: "pond" }],
};

const map = P.generateMap(SPEC);
const occupied = () => [
  ...Object.values(map.npcs),
  ...Object.values(map.items),
];

function finished() {
  const s = W.initialWorldState(SPEC);
  return { ...s, completedQuests: ["lost_rod"] };
}

test("director runs when no quest is left, or right after one is finished", () => {
  const s = W.initialWorldState(SPEC);
  assert.equal(D.directorDue(SPEC, s, false), false);
  assert.equal(D.directorDue(SPEC, s, true), true);
  assert.equal(D.directorDue(SPEC, finished(), false), true);
  // Three open quests is enough.
  const busy = {
    ...SPEC,
    quests: [0, 1, 2].map((i) => ({ ...SPEC.quests[0], id: `q${i}` })),
  };
  assert.equal(D.directorDue(busy, s, true), false);
});

test("a patch adds a quest, item and villager on reachable free tiles, and the map stays the same", () => {
  const tiles = map.tiles.slice();
  const out = D.applyPatch(SPEC, map, finished(), PATCH, occupied());
  assert.ok(out);
  assert.deepEqual(map.tiles, tiles);
  const reach = P.reachable(map, map.spawn);
  for (const p of [...out.npcs, ...out.items]) {
    assert.equal(P.isSolid(map, p.x, p.y), false);
    assert.equal(reach[p.y * map.width + p.x], 1);
    for (const o of occupied())
      assert.ok(Math.abs(o.x - p.x) + Math.abs(o.y - p.y) > 1);
  }
  // Inside the region asked for when it has room.
  const it = out.items[0];
  assert.ok(it.x >= 2 && it.x < 10 && it.y >= 2 && it.y < 8);
  assert.deepEqual(out.state.activeQuests, ["ring_bell"]);
  const played = D.withAdditions(SPEC, out.state.additions);
  assert.deepEqual(
    played.quests.map((q) => q.id),
    ["lost_rod", "ring_bell"],
  );
});

test("placement never picks a tile the player cannot reach or one next to something placed", () => {
  // 12x12, open ground inside a wall border, with a walled pocket at x 6..8, y 6..8 (only 7,7 is open).
  const w = 12;
  const grass = P.TILE_KINDS.indexOf("grass");
  const wall = P.TILE_KINDS.indexOf("wall");
  const tiles = new Uint8Array(w * w).fill(grass);
  for (let i = 0; i < w; i++) {
    tiles[i] =
      tiles[(w - 1) * w + i] =
      tiles[i * w] =
      tiles[i * w + w - 1] =
        wall;
  }
  for (let y = 6; y <= 8; y++)
    for (let x = 6; x <= 8; x++)
      if (x !== 7 || y !== 7) tiles[y * w + x] = wall;
  const m = {
    width: w,
    height: w,
    tiles,
    spawn: { x: 2, y: 2 },
    npcs: {},
    items: {},
  };
  const taken = [{ x: 4, y: 7 }];
  const area = { x: 3, y: 6, w: 6, h: 3 };
  for (let seed = 0; seed < 200; seed++) {
    const p = D.placeFree(m, taken, area, seed);
    assert.ok(p);
    assert.notDeepEqual(p, { x: 7, y: 7 });
    assert.ok(Math.abs(p.x - 4) + Math.abs(p.y - 7) > 1);
    assert.ok(p.x >= 3 && p.x < 9 && p.y >= 6 && p.y < 9);
  }
});

test("patches with dangling or reused ids, or past a limit, are dropped", () => {
  const s = finished();
  const bad: Patch[] = [
    { ...PATCH, quest: { ...PATCH.quest, id: "lost_rod" } },
    { ...PATCH, quest: { ...PATCH.quest, giver: "nobody" } },
    { ...PATCH, quest: { ...PATCH.quest, requires_item: "teapot" } },
    { ...PATCH, items: [{ id: "rod", name: "Rod 2", description: "" }] },
    {
      ...PATCH,
      npcs: [{ id: "tom", name: "Tom 2", persona: "", greeting: "" }],
    },
  ];
  for (const p of bad)
    assert.equal(D.applyPatch(SPEC, map, s, p, occupied()), null);
  const full = {
    ...s,
    additions: {
      ...D.noAdditions(),
      npcs: [0, 1, 2, 3].map((i) => ({
        id: `n${i}`,
        name: `N${i}`,
        persona: "",
        greeting: "",
        x: 3,
        y: 3,
      })),
    },
  };
  assert.equal(D.applyPatch(SPEC, map, full, PATCH, occupied()), null);
});

test("additions survive save and load, and keep their quests and items", () => {
  const out = D.applyPatch(SPEC, map, finished(), PATCH, occupied());
  assert.ok(out);
  const carried = { ...out.state, inventory: ["bell"], picked: ["bell"] };
  const back = W.deserialize(
    SPEC,
    JSON.parse(JSON.stringify(W.serialize(carried))),
  );
  assert.ok(back);
  assert.deepEqual(back.additions, out.state.additions);
  assert.deepEqual(back.activeQuests, ["ring_bell"]);
  assert.deepEqual(back.inventory, ["bell"]);
  // A save from before additions loads with none.
  const old = W.deserialize(SPEC, {
    ...W.serialize(W.initialWorldState(SPEC)),
    additions: undefined,
  });
  assert.deepEqual(old?.additions, D.noAdditions());
  // Junk in a save is dropped: a quest whose giver is gone, an item off the map.
  const junk = D.additionsOf(
    {
      quests: [{ ...PATCH.quest, giver: "ghost" }],
      items: [{ id: "x", name: "X", description: "", x: 999, y: 1 }],
    },
    SPEC,
  );
  assert.deepEqual(junk, D.noAdditions());
});

test("an added villager's mind survives save and load", () => {
  const sim = {
    ...SPEC,
    simulation: {
      enabled: true,
      minutes_per_second: 2,
      start_minute: 480,
      places: [{ id: "pond", name: "Pond", x: 2, y: 2, w: 8, h: 6 }],
      chat_cooldown_min: 120,
      calls_per_minute: 4,
    },
  } as unknown as Spec;
  const out = D.applyPatch(sim, map, finished(), PATCH, occupied());
  assert.ok(out);
  const mind = W.initialWorldState({
    ...sim,
    npcs: [...sim.npcs, PATCH.npcs?.[0]],
  } as Spec).npc.mira;
  assert.ok(mind);
  const remembered = {
    ...out.state,
    npc: {
      ...out.state.npc,
      mira: {
        ...mind,
        memory: [
          {
            id: "m0",
            text: "Pip rang the bell",
            kind: "observation",
            t_created: 480,
            t_access: 480,
            importance: 5,
          },
        ],
      },
    },
  };
  const back = W.deserialize(
    sim,
    JSON.parse(JSON.stringify(W.serialize(remembered as never))),
  );
  assert.equal(back?.npc.mira?.memory.length, 1);
});

test("the player keeps the map from the generated world and adds through the game handle", () => {
  const src = readFileSync(
    new URL(
      "../src/features/flows/player/world/world-player.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(src, /generateMap\(baseSpec\)/);
  assert.match(src, /h\?\.addNpc\(n, n\)/);
  assert.match(src, /h\?\.addItem\(i, i\)/);
  assert.match(src, /deserialize\(baseSpec, rec\.state\)/);
  // The director reads the state the reply just produced, not one React has yet to commit.
  assert.match(src, /await direct\(completed, state\)/);
  assert.match(src, /const s = from \?\? wsRef\.current;/);
});
